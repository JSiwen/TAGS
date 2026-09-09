def linear_warmup(iteration, start, end, init_value, final_value):
    if end <= start:
        return final_value
    if iteration <= start:
        return init_value
    if iteration >= end:
        return final_value
    alpha = float(iteration - start) / float(end - start)
    return (1.0 - alpha) * init_value + alpha * final_value


def linear_warmup_decay(iteration, warmup_start, warmup_end, decay_start, decay_end, init_value, peak_value, final_value):
    if iteration <= warmup_end:
        return linear_warmup(iteration, warmup_start, warmup_end, init_value, peak_value)
    if iteration <= decay_start:
        return peak_value
    return linear_warmup(iteration, decay_start, decay_end, peak_value, final_value)
