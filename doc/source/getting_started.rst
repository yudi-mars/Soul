快速入门
==============

为了快速了解如何利用Soul训练一个SNN模型，让我们从一个代码案例开始，模型训练整个过程被分为：

* 环境准备
* 训练准备
* 模型训练
* 数据测量

四个阶段展开。


环境准备
----------------
首先，我们需要为我们的训练过程配置适当的参数：

.. code-block:: python

    config = init_config()
    
``init_config()`` 是 ``soul.utils`` 包下的函数，它会读取我们提前设置的配置信息并生成配置对象。
配置信息的来源主要有两个：

* ``parse_args()`` 方法：在 ``soul.utils.parser.parse_args()`` 方法中，我们可以根据其中参数设置诸如：数据集地址、时间步、训练批次大小等训练基础信息
* config文件夹中的配置文件：在 ``soul/config/`` 文件夹中包含大量的配置文件，主要分为：

    * ``config/basic.yaml`` 基础配置文件
    * ``config/neuron/*.yaml`` 不同神经元的特殊配置
    * ``config/model/application/*.yaml`` 不同模态模型的特殊配置
    
配置的优先级为： ``config/model/application/*.yaml`` > ``config/neuron/*.yaml`` > ``parse_args()`` > ``config/basic.yaml``
更多关于基础配置（ ``parse_args()`` 、 ``config/basic.yaml`` ）的详细信息，请参考 :doc:`../concepts`
关于神经元和不同模态模型的特殊配置，请分别参考 :doc:`../neuron` 和 :doc:`../neuron`

在这之后，Soul会检查 ``RANK`` 和 ``WORLD_SIZE`` 环境变量，判断是否进入分布式模式,若处于分布式模式，则初始化进程间通信并绑定GPU：

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

在本案例中，优先使用GPU进行模型的训练
Soul会为您准备logger以备调试信息的输出：

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

训练准备
----------------
在训练准备阶段，我们首先要准备训练所需的训练数据和测试：

.. code-block:: python

    train_dataset, test_dataset = load_dataset(config)

``load_dataset(config)`` 是 ``soul.utils`` 包下的函数，他会读取之前准备的配置对象中的信息，并按照要求读取和处理数据集的数据，需要注意的是，
训练集的数据需要提前下载至本地供Soul使用，Soul使用的数据集及其下载地址可从 `Soul <https://github.com/yudi-mars/Soul>`_ 上获得。
有关数据集的配置信息，请参考 :doc:`../datasets/index`

对于分布式训练的训练集，我们需要定义采样器：

.. code-block:: python

    if config['is_distributed']:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        # define the batch size per gpu, usually we define the numer of process equal to the number of used gpus
        world_size = dist.get_world_size()
        config['batch_size'] //= world_size
    else:
        train_sampler = None

构建训练集与测试集的DataLoader：

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

我们还需要使用配置对象中的信息来指定训练使用的梯度代理函数及神经元类型，之后Soul将使用配置对象中的信息构建其我们需要训练的模型：

.. code-block:: python

    config['surrogate_function'] = surrogate_map[config['surrogate']]
    config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 
    model = model_map[config['application']][config['model'].lower()](config)

其中，您将使用的神经元 ``config['neuron_type']`` 及我们需要训练的模型类型 ``config['model']`` 可以根据 ``soul.utils.parser.parse_args()`` 中的对应参数被配置
``config['surrogate']`` 由您选定的神经元的特点配置 ``config/neuron/*.yaml`` 所决定
``config['application']`` 由我们使用的数据集自动对应，无需手动设置

通过 ``soul.utils.metrics.num_params.count_parameters(model, trainable=False)`` 方法，我们可以查看模型的参数量是多少：

.. code-block:: python

    if global_rank == 0:
        n_parameters = count_parameters(model, trainable=True) 
        logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

将模型传输至指定设备上（本例中优先使用GPU），若使用分布式训练，则需要使用 ``torch.nn.parallel.distributed.DistributedDataParallel`` 类对模型进行包装：

.. code-block:: python

    model.to(device)
    if config['is_distributed']:
        model = DDP(model, device_ids=[local_rank])

在训练准备阶段结束之前，我们还需要初始化损失函数、优化器及训练调度器的信息：

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

本案例中以交叉熵损失作为损失函数， ``config['optimizer']`` 及 ``config['scheduler']`` 可以根据 ``soul.utils.parser.parse_args()`` 方法中的对应参数进行配置
关于优化器及训练调度器的配置的更多信息，请参考 :doc:`../params`


模型训练
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

``AverageMeter()`` 来源于 ``soul.utils.metrics.AverageMeter`` ，用于计算和存储训练中的结果
``accuracy(output, target, topk=(1,))`` 来源于 ``soul.utils.metrics.accuracy(output, target, topk=(1,))`` ，用于计算准确率
在该案例中，代码会保存当前测试集准确率最佳的模型参数到地址 ``config['model_dir']`` ，可以根据 ``soul.utils.parser.parse_args()`` 方法中的对应参数进行配置


数据测量
----------------
在 ``soul.utils.metrics`` 包下，我们提供了多种测试模型性能的工具，本案例以 ``soul.utils.metrics.sops`` 为例，该工具用于测量理论能量消耗

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

其中 ``config['sop']`` 、 ``config['e_ac']`` 、 ``config['e_mac']`` 分别表示是否模拟在神经形态芯片上的能量消耗、在假定的 45 纳米硬件上，累加操作运行的能耗成本/皮焦（pJ）和在假定的 45 纳米硬件上，乘法累加操作的能耗成本/皮焦（pJ）
，这些参数可以在 ``config/basic.yaml`` 中被配置，更多信息请参考 :doc:`../params`