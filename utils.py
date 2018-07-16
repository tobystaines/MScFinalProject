def partial_argv(func, *args, **kwargs):
    '''
    Parameters
    ----------
    func    : A function that takes scalar argument and returns scalar value
    *args   : Args to partially apply to func
    *kwargs : Keyword args to partially apply to func

    Returns
    -------
    func(*args) : A function that maps func over *args and returns a tuple

    Example
    -------
    func = partial_argv(abs)
    func(-1, 2, -3, 4)

    # returns (1, 2, 3, 4)
    '''
    return lambda *other_args: tuple(map(partial(func, *args, **kwargs), other_args))


def zip_tensor_slices(*args):
    '''
    Parameters
    ----------
    *args : list of _n_ _k_-dimensional tensors, where _k_ >= 2
        The first dimension has _m_ elements.

    Returns
    -------
    result : Dataset of _m_ examples, where each example has _n_
        records of _k_ - 1 dimensions.

    Example
    -------
    ds = (
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensors([[1,2], [3,4], [5, 6]]),
            tf.data.Dataset.from_tensors([[10, 20], [30, 40], [50, 60]])
        ))
        .flat_map(zip_tensor_slices)  # <--- *HERE*
    )
    el = ds.make_one_shot_iterator().get_next()
    print sess.run(el)
    print sess.run(el)

    # Output:
    # (array([1, 2], dtype=int32), array([10, 20], dtype=int32))
    # (array([3, 4], dtype=int32), array([30, 40], dtype=int32))
    '''
    return tf.data.Dataset.zip(tuple([
        tf.data.Dataset.from_tensor_slices(arg)
        for arg in args
    ]))