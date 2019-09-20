
import torch
import time
from lib.utils.metric_logger import MetricLogger
from lib.engine.eval import evaluate
from torch.nn.utils import clip_grad_norm_
import datetime

def train(
    model,
    optimizer,
    dataloader,
    device,
    params,
    checkpointer=None,
    tensorboard=None,
    getter=None,
    dataloader_val=None,
    evaluator=None
):
    """
    :param params:
        - max_epochs: required
        - checkpoint_period: if None, after every epoch
        - print_every
        - val_every
    """
    # get parameters
    max_epochs = params.get('max_epochs')
    print_every = params.get('print_every', 100)
    # val_every = params.get('val_every', 100)
    checkpoint_period = params.get('checkpoint_period')

    # start from where we left
    start_epoch = 0
    if checkpointer:
        start_epoch = checkpointer.args['epoch']

    max_iter = len(dataloader) * max_epochs

    meters = MetricLogger(', ')

    print('Start training')
    for epoch in range(start_epoch, max_epochs):
        epoch = epoch + 1

        for idx, data in enumerate(dataloader):
            model.train()

            # Note, first one is image. This is not neat. Just for convenience.
            data = data[0]
            batch_size = data.size()[0]

            if idx > len(dataloader):
                break
            idx = idx + 1
            global_iter = epoch * len(dataloader) + idx

            start_time = time.perf_counter()
            data = data.to(device)
            loss = model(data)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_time = time.perf_counter() - start_time
            loss= loss.item()
            meters.update(loss=loss)
            meters.update(batch_time=batch_time)

            # display logs
            if idx % print_every == 0:
                # we will compute estimated time
                eta = (max_iter - global_iter) * meters['batch_time'].global_avg
                # print(max_iter, global_iter, batch_time)
                eta = datetime.timedelta(seconds=int(eta))
                print(meters.delimiter.join([
                    'eta: {eta}',
                    'epoch: {epoch}',
                    'iter: {idx}',
                    'loss: {loss:.4f}',
                    'batch-time: {batch_time:.4f}s',
                    'lr: {lr}'
                ]).format(
                    eta=eta,
                    epoch=epoch,
                    idx=idx,
                    loss=meters['loss'].median,
                    batch_time=meters['batch_time'].median,
                    lr=optimizer.param_groups[0]['lr']
                ))

                # tensorboard
                tb_data = getter.get_tensorboard_data()
                if not tensorboard is None:
                    tensorboard.update(var=model.sigma)
                    tensorboard.update(loss=meters['loss'].median)
                    tensorboard.update(time_per_iteration=meters['batch_time'].global_avg / batch_size)
                    tensorboard.update(med_time_per_iteration=meters['batch_time'].median / batch_size)
                    tensorboard.update(**tb_data)
                    tensorboard.add('train', global_iter)

                with torch.no_grad():
                    model.eval()
                    data = next(iter(dataloader_val))
                    data = data[0]
                    data = data.to(device)
                    model(data)

                    # tensorboard
                    tb_data = getter.get_tensorboard_data()
                    if not tensorboard is None:
                        tensorboard.update(var=model.sigma)
                        tensorboard.update(loss=meters['loss'].median)
                        tensorboard.update(time_per_iteration=meters['batch_time'].global_avg / batch_size)
                        tensorboard.update(med_time_per_iteration=meters['batch_time'].median / batch_size)
                        tensorboard.update(**tb_data)
                        tensorboard.add('valid', global_iter)

            # checkpoint
            if (idx + 1) % checkpoint_period == 0 and checkpointer is not None:
                checkpointer.args['epoch'] = epoch
                checkpointer.args['iter'] = idx
                checkpointer.save('model_{:04d}_{:06d}'.format(epoch, idx))
