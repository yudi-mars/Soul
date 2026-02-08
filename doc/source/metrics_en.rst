Metric Tools
========

In this chapter, we will learn how to use each utility file in the ``soul.utils.metrics`` package.

accuracy.py
----------------
* AverageMeter：A utility class for counting and storing training results, used as follows:

.. code-block:: python

    # Define utility objects to record Top1 accuracy and loss value respectively
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
    # Obtain the average values of accuracy and loss via the avg attribute
    test_acc = test_top1_meter.avg
    test_loss = test_loss_meter.avg

* accuracy(output, target, topk=(1,))：Calculates the accuracy of the top-k predicted results for the specified topk values, used as shown above.

num_params.py
----------------
* count_parameters(model, trainable=False)：Helps you count the number of parameters in an SNN model, used as follows:

.. code-block:: python

    n_parameters = count_parameters(model, trainable=True) 
    logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

sops.py
----------------
This utility counts theoretical energy consumption by registering hook functions, used as follows:

.. code-block:: python

    # calculate theoretical energy cost per sample inference
    logger.info('Counting FLOPs/SOPs for theoretical inference cost')
    # Define an ops monitor; the is_sop parameter indicates whether to simulate computation on neuromorphic chips
    ops_monitor(model, is_sop=config['sop'])
    for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Count OPs: '):
        # default data shape (B, T, input_size) -> (T, B, input_size)
        inputs = inputs.transpose(0, 1).to(device)
        _ = model(inputs)

    # Get the number of numerical operations for each Module
    total_sops = 0
    for k, v in MODULE_SOP_DICT.items():
        total_sops += v
    avg_sops = total_sops / len(test_loader)

    # cost_per_op represents the energy consumption per computational operation
    cost_per_op = config['e_ac'] if config['sop'] else config['e_mac']
    logger.info(f"Average number of {'SOPs' if config['sop'] else 'FLOPs'} for model {config['model']} inference per sample: {avg_sops / 1e6:.2f} M")
    logger.info(f"corresponding theoretical energy cost: {avg_sops * cost_per_op / 1e9:.2f} mj")

power_energy.py
----------------
An implementation scheme for energy consumption monitoring when running Spiking Neural Networks (SNNs) on Jetson series edge devices, used as follows:

.. code-block:: python

    # Get the current device
    device = getDevice()
    print(f"Detected device: {device}")  # Print the device name

    # Define an energy consumption monitor; interval indicates the sampling interval
    pl = PowerLogger(interval=0.05)
    # Start monitoring
    pl.start()
    time.sleep(5)
    print('5s IDLE time passed, start IO bench mark now!')

    # Insert model inference code here!

    # Event information can be recorded via the recordEvent(str) method
    pl.recordEvent('started IO bench mark')
    time.sleep(2)
    pl.recordEvent('ding! 3s')
    # Simulate computational bottlenecks with stress command
    os.system('stress -c 12 -t 3')
    time.sleep(1.5)
    pl.recordEvent('ding! 2s')
    os.system('stress -c 1 -t 2')
    time.sleep(2)
    pl.recordEvent('ding! 1s')
    os.system('stress -c 2 -t 1 -m 4')
    time.sleep(1.5)
    # Stop monitoring
    pl.stop()
    # Print total energy consumption
    pl.showDataTraces()
    # Print a complete report of all monitoring nodes, e.g., (power,voltage,current)
    print(printFullReport(getDevice()))
    