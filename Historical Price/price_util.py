
import numpy as np

def ohlc_aggregate(price: np.ndarray, interval: int) -> np.ndarray:
    """
    Input:
    - price: shape (N*interval, 4)
    - interval: interval to aggregate. ex: minute->hour is 60, hour->day is 24.
    Output:
    - aggr_price: shape (N, 4)
    """
    price = price.reshape((-1, interval, 4))
    o, h, l, c = np.split(price, 4, axis=-1)
    aggr_price = np.concatenate([
        o[:, 0],
        h.max(axis=1),
        l.min(axis=1),
        c[:, -1]
    ], axis=-1)
    return aggr_price

def ohlc_price_to_delta(price: np.ndarray, day_price: np.ndarray) -> np.ndarray:
    """
    Input:
    - price: shape (N*n_days, 4)
    - day_price: shape (n_days, 4)
    Output:
    - delta: shape (N*n_days, 4)
    """
    price = price.reshape((day_price.shape[0], -1, 4))
    day_price = np.concatenate([[day_price[0, 0]], day_price[:-1, -1]])[:, np.newaxis, np.newaxis]
    delta = price / day_price - 1
    return delta.reshape((delta.shape[0]*delta.shape[1], 4))

def ohlc_delta_to_price(delta: np.ndarray, day_price: np.ndarray) -> np.ndarray:
    """
    Input:
    - delta: shape (N*n_days, 4)
    - day_price: shape (n_days, 4)
    Output:
    - price: shape (N*n_days, 4)
    """
    delta = delta.reshape((day_price.shape[0], -1, 4))
    day_price = np.concatenate([[day_price[0, 0]], day_price[:-1, -1]])[:, np.newaxis, np.newaxis]
    price = (delta + 1) * day_price
    return price.reshape((price.shape[0]*price.shape[1], 4))

if __name__ == '__main__':
    minute_price = np.tile(np.arange(1, 2*24*60+1)[:, np.newaxis], (1, 4))
    hour_price = ohlc_aggregate(minute_price, 60)
    day_price = ohlc_aggregate(hour_price, 24)
    print(hour_price)
    print(day_price)
    print(ohlc_price_to_delta(hour_price, day_price))
    print(ohlc_price_to_delta(day_price, day_price))
    print(np.isclose(ohlc_delta_to_price(ohlc_price_to_delta(hour_price, day_price), day_price), hour_price).all())
