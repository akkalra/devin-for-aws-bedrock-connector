import pytest

from batching import bucket_by_length, pad_batch


def test_pad_batch_pads_to_longest():
    ids, mask = pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert ids == [[1, 2, 3], [4, 0, 0]]
    assert mask == [[1, 1, 1], [1, 0, 0]]


def test_bucket_exact_fit_goes_in_that_bucket():
    buckets = bucket_by_length([[1] * 128, [1] * 5], [128, 512])
    assert len(buckets[128]) == 2
    assert buckets[512] == []


def test_bucket_too_long_raises():
    with pytest.raises(ValueError):
        bucket_by_length([[1] * 600], [128, 512])
