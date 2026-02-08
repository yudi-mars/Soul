度量工具
========

本章，我们将了解 ``soul.utils.metrics`` 包中各个工具文件的使用方法。

accuracy.py
----------------
* AverageMeter：统计和存储训练结果的工具类，使用方法如下：

.. code-block:: python

    # 定义工具对象分别记录Top1准确率和损失值
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
    # 通过avg属性获得准确率和损失值的平均值
    test_acc = test_top1_meter.avg
    test_loss = test_loss_meter.avg

* accuracy(output, target, topk=(1,))：针对指定的topk值，计算前topk个预测结果的准确率，使用方法如上。

num_params.py
----------------
* count_parameters(model, trainable=False)：帮助您计算SNN模型的参数量，使用方法如下：

.. code-block:: python

    n_parameters = count_parameters(model, trainable=True) 
    logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

sops.py
----------------
该工具通过注册钩子函数统计理论能量消耗，使用方法如下：

.. code-block:: python

    # calculate theoretical energy cost per sample inference
    logger.info('Counting FLOPs/SOPs for theoretical inference cost')
    # 定义ops监视器，is_sop参数表示是否模拟在神经形态芯片上的计算情况
    ops_monitor(model, is_sop=config['sop'])
    for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Count OPs: '):
        # default data shape (B, T, input_size) -> (T, B, input_size)
        inputs = inputs.transpose(0, 1).to(device)
        _ = model(inputs)

    # 取得各个Module的数值操作数
    total_sops = 0
    for k, v in MODULE_SOP_DICT.items():
        total_sops += v
    avg_sops = total_sops / len(test_loader)

    # cost_per_op表示每个计算操作的能耗为多少
    cost_per_op = config['e_ac'] if config['sop'] else config['e_mac']
    logger.info(f"Average number of {'SOPs' if config['sop'] else 'FLOPs'} for model {config['model']} inference per sample: {avg_sops / 1e6:.2f} M")
    logger.info(f"corresponding theoretical energy cost: {avg_sops * cost_per_op / 1e9:.2f} mj")

power_energy.py
----------------
在 Jetson 系列边缘设备上运行脉冲神经网络（SNNs）时，能耗监测的实现方案，使用方法如下：

.. code-block:: python

    # 获取当前设备
    device = getDevice()
    print(f"Detected device: {device}")  # 打印出设备名称

    # 定义能耗监视器，interval表示采样间隔
    pl = PowerLogger(interval=0.05)
    # 开始监视
    pl.start()
    time.sleep(5)
    print('5s IDLE time passed, start IO bench mark now!')

    # 插入模型推理代码！

    # 可以通过 recordEvent(str) 方法记录事件信息
    pl.recordEvent('started IO bench mark')
    time.sleep(2)
    pl.recordEvent('ding! 3s')
    # 通过stress模拟计算瓶颈
    os.system('stress -c 12 -t 3')
    time.sleep(1.5)
    pl.recordEvent('ding! 2s')
    os.system('stress -c 1 -t 2')
    time.sleep(2)
    pl.recordEvent('ding! 1s')
    os.system('stress -c 2 -t 1 -m 4')
    time.sleep(1.5)
    # 监视结束
    pl.stop()
    # 打印总能耗
    pl.showDataTraces()
    # 打印所有监控节点的完整报告，如(power,voltage,current)
    print(printFullReport(getDevice()))
    