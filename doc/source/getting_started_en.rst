Starting with an Example
==============

To quickly learn how to use SOUL to train an SNN model, let’s start with a code example. The entire model training process is divided into four stages:

* Environment Preparation
* Training Preparation
* Model Training
* Data Measurement


Environment Preparation
----------------
First, we need to configure appropriate parameters for our training process:

.. code-block:: python

    config = init_config()
    
``init_config()`` is a function under the soul.utils package. It reads the pre-configured settings and generates a configuration object.
There are two main sources of configuration information:

* ``parse_args()`` method: In the ``soul.utils.parser.parse_args()`` method, we can set basic training information such as dataset path, time steps, and training batch size through its parameters.
* Configuration files in the config folder: The ``soul/config/`` folder contains a large number of configuration files, mainly categorized into:

    * ``config/basic.yaml`` Basic configuration file
    * ``config/neuron/*.yaml`` Special configurations for different neurons
    * ``config/model/application/*.yaml`` Special configurations for models of different modalities
    
The priority of configurations is:  ``config/model/application/*.yaml`` > ``config/neuron/*.yaml`` > ``parse_args()`` > ``config/basic.yaml``
For more detailed information about basic configurations （ ``parse_args()`` 、 ``config/basic.yaml`` ），please refer to  :doc:`../concepts`
For special configurations of neurons and models of different modalities, please refer to :doc:`../neuron` and :doc:`../neuron`

After this, Soul checks the ``RANK`` and ``WORLD_SIZE`` environment variables to determine whether to enter distributed mode. If in distributed mode, it initializes inter-process communication and binds the GPU:

.. code-block:: python

    config['is_distributed'] = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if config['is_distributed']:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # gpu for current process
        device = torch.device("cuda", local_rank)
        # main process
        global_rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        global_rank = 0

In this example, GPU is preferred for model training.
Soul will prepare a logger for outputting debugging information:

.. code-block:: python

    if global_rank == 0:
        log_path = os.path.join(
            config['log_dir'], 
            config['dataset_name'].lower(), 
            config['model'].lower(), 
            config['neuron_type'].lower()
        )
        ensure_dir(log_path)
        logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])
        logger.info(f'Distributed Training: {config["is_distributed"]}')
    else:
        logger = None

Training Preparation
----------------
In the training preparation stage, we first need to prepare the training and test data required for training:

.. code-block:: python

    train_dataset, test_dataset = load_dataset(config)

``load_dataset(config)`` is a function under the ``soul.utils`` package. It reads information from the previously prepared 
configuration object and loads/processes the dataset data as required. Note that the training set data needs to be downloaded 
locally in advance for Soul to use. The datasets used by Soul and their download links can be obtained from 
`Soul <https://github.com/yudi-mars/Soul>`_ 。
For configuration information about datasets, please refer to :doc:`../datasets/index`

For the training set in distributed training, we need to define a sampler:

.. code-block:: python

    if config['is_distributed']:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        # define the batch size per gpu, usually we define the numer of process equal to the number of used gpus
        world_size = dist.get_world_size()
        config['batch_size'] //= world_size
    else:
        train_sampler = None

For the training set in distributed training, we need to define a sampler:

.. code-block:: python

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle= False if config['is_distributed'] else True,
        sampler=train_sampler, 
        num_workers=config['workers'], 
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['workers'], 
        pin_memory=True
    )

We also need to use the information in the configuration object to specify the gradient surrogate function and neuron type for training. After that, 
Soul will use the information in the configuration object to build the model we need to train:

.. code-block:: python

    config['surrogate_function'] = surrogate_map[config['surrogate']]
    config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 
    model = model_map[config['application']][config['model'].lower()](config)

Among them, the neuron you will use ``config['neuron_type']`` and the type of model we need to train ``config['model']`` can be configured through the corresponding parameters in ``soul.utils.parser.parse_args()`` 中的对应参数被配置
``config['surrogate']`` is determined by the ``config/neuron/*.yaml`` configuration selected based on the characteristics of the neuron you choose.
``config['application']`` is automatically mapped based on the dataset we use and does not require manual setting.

We can check the number of parameters of the model using the ``soul.utils.metrics.num_params.count_parameters(model, trainable=False)`` method:

.. code-block:: python

    if global_rank == 0:
        n_parameters = count_parameters(model, trainable=True) 
        logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

Transfer the model to the specified device (GPU is preferred in this example). If using distributed training, the model needs to be wrapped using the ``torch.nn.parallel.distributed.DistributedDataParallel`` class:

.. code-block:: python

    model.to(device)
    if config['is_distributed']:
        model = DDP(model, device_ids=[local_rank])

Before the end of the training preparation stage, we also need to initialize the loss function, optimizer, and training scheduler:

.. code-block:: python

    criterion = nn.CrossEntropyLoss()
    # init optimzer
    if config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        if global_rank == 0:
            logger.warning(f"Received unrecognized optimizer {config['optimizer']}, set default Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # init scheduler
    if config['scheduler'].lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config['scheduler'].lower() == 'linear':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["epochs"] * 0.25), gamma=0.1)
    elif config['scheduler'].lower() == 'warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(config["epochs"] * 0.1), T_mult=2)
    else:
        if global_rank == 0:
            logger.warning(f"Received unrecognized scheduler {config['scheduler']}, set default ConsineAnnealing Scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

In this example, cross-entropy loss is used as the loss function.  ``config['optimizer']`` and ``config['scheduler']`` can be configured through the corresponding parameters in the ``soul.utils.parser.parse_args()`` method.
For more information about the configuration of optimizers and training schedulers, please refer to :doc:`../params`


Model Training
----------------

.. code-block:: python

    best_acc = 0.
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        if config['is_distributed']:
            train_sampler.set_epoch(epoch)
        
        train_top1_meter, train_loss_meter = AverageMeter(), AverageMeter()
        # customize progress bar for train loader
        loader = tqdm(train_loader, unit='batch', ncols=80, desc='Train: ') if global_rank == 0 else train_loader
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            # default data shape (B, T, input_size) -> (T, B, input_size)
            inputs = inputs.transpose(0, 1)

            outputs = model(inputs)
            acc1 = accuracy(outputs, targets, topk=(1,))[0]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_top1_meter.update(acc1.item(), targets.numel())
            train_loss_meter.update(loss.item(), targets.numel())

        train_acc = train_top1_meter.avg
        train_loss = train_loss_meter.avg

        if not config['is_distributed'] or dist.get_rank() == 0:
            model.eval()

            test_top1_meter, test_loss_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                    # default data shape (B, T, input_size) -> (T, B, input_size)
                    inputs = inputs.transpose(0, 1)

                    outputs = model(inputs)
                    acc1 = accuracy(outputs, targets, topk=(1,))[0]
                    loss = criterion(outputs, targets)

                    test_loss_meter.update(loss.item(), targets.numel())
                    test_top1_meter.update(acc1.item(), targets.numel())

            test_acc = test_top1_meter.avg
            test_loss = test_loss_meter.avg

            logger.info(f"[Epoch {epoch:3d}/{config['epochs']}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%; Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            if test_acc > best_acc:
                ensure_dir(config['model_dir'])

                best_acc = test_acc
                logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
                torch.save(
                    model.module.state_dict() if config['is_distributed'] else model.state_dict(), 
                    os.path.join(
                        config['model_dir'], 
                        f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_T{config["time_step"]}_{config["seed"]}.pt'
                    )
                )
        scheduler.step()

``AverageMeter()``  is from  ``soul.utils.metrics.AverageMeter`` and is used to calculate and store results during training.
``accuracy(output, target, topk=(1,))`` is from ``soul.utils.metrics.accuracy(output, target, topk=(1,))`` and is used to calculate accuracy.
In this example, the code saves the model parameters with the best current test set accuracy to the path ``config['model_dir']`` , which can be configured through the corresponding parameters in the ``soul.utils.parser.parse_args()`` method.


Data Measurement
----------------
Under the ``soul.utils.metrics`` package, we provide a variety of tools for testing model performance. This example uses ``soul.utils.metrics.sops`` , which is used to measure theoretical energy consumption.

.. code-block:: python

    # monitor max memory footprint with the best model
    if not config['is_distributed'] or dist.get_rank() == 0:
        best_model_path = os.path.join(config['model_dir'], f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
        logger.info(f'The size of model parameter checkpoint file: {os.path.getsize(best_model_path) / (1024 ** 2):.2f} MB')
        best_params = torch.load(
            best_model_path, 
            map_location='cpu', 
            weights_only=True
        )
        if config['is_distributed']:
            model.module.load_state_dict(best_params)
        else:
            model.load_state_dict(best_params)
        logger.debug(f'current device to monitor: {device}')
        model.to(device)
        model.eval()

        # calculate theoretical energy cost per sample inference
        logger.info('Counting FLOPs/SOPs for theoretical inference cost')
        ops_monitor(model, is_sop=config['sop'])
        for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Count OPs: '):
            # default data shape (B, T, input_size) -> (T, B, input_size)
            inputs = inputs.transpose(0, 1).to(device)
            _ = model(inputs)

        total_sops = 0
        for k, v in MODULE_SOP_DICT.items():
            total_sops += v
        avg_sops = total_sops / len(test_loader)

        cost_per_op = config['e_ac'] if config['sop'] else config['e_mac']
        logger.info(f"Average number of {'SOPs' if config['sop'] else 'FLOPs'} for model {config['model']} inference per sample: {avg_sops / 1e6:.2f} M")
        logger.info(f"corresponding theoretical energy cost: {avg_sops * cost_per_op / 1e9:.2f} mj")

Among them, ``config['sop']`` 、 ``config['e_ac']`` 、 ``config['e_mac']`` respectively indicate whether to simulate energy consumption on neuromorphic chips, the energy consumption cost per accumulation operation (in picojoules, pJ) on hypothetical 45nm hardware, and the energy consumption cost per multiply-accumulate operation (in picojoules, pJ) on hypothetical 45nm hardware. These parameters can be configured in ``config/basic.yaml`` . 
For more information, please refer to :doc:`../params`