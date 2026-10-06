def show_pred(batch_size, net, device, source):
    """Run conditional sampling for a batch of layout images."""
    return net.sample(batch_size, device, source)


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
