from functools import wraps
GIB=1024**3
def with_fea_reserve(frozen_check):
    @wraps(frozen_check)
    def check(sample, *, phase, ready_events=None):
        frozen_check(sample,phase=phase,ready_events=ready_events)
        assert sample['host']['memavailable'] >= 64*GIB, 'FEA global reserve <64GiB'
    return check
