
import pandas as pd
import json
import math
import pytz
import datetime
import collections
import threading
import queue
import multiprocessing as mp
import snscrape.modules.twitter as sntwitter

def scrape_multithread(queries, dates, max_threads):
    thread_lst = []
    stop_event = threading.Event()
    work_queue = queue.Queue()
    for query, date in zip(queries, dates):
        work_queue.put((query, date))
    for i in range(max_threads):
        thread_lst.append(
            threading.Thread(
                target=scrape_task, 
                args=(work_queue, stop_event)
            )
        )
        thread_lst[i].start()
    work_queue.join()
    stop_event.set()
    for t in thread_lst:
        t.join()

def scrape_task(q, stop_event):
    while not stop_event.is_set():
        try:
            query, date = q.get(timeout=0.5)
            scrape(query, date)
        except queue.Empty:
            continue
        q.task_done()

def scrape(query, date):
    date = pd.Timestamp(date)
    next_day = date + pd.Timedelta(days=2)
    scraper = sntwitter.TwitterSearchScraper(f'{query} since:{date.strftime("%Y-%m-%d")} until:{next_day.strftime("%Y-%m-%d")}')
    fname = f'./tweets{query}/{date.year}/{date.month:02}/{date.strftime("%Y-%m-%d")}-tweets-{query}-compressed.csv'
    start = pd.Timestamp.now()
    print(f'Date: {date.strftime("%Y-%m-%d")} | Beg: {start.strftime("%Y-%m-%d %H:%M:%S")}')
    tweets, header, mode = [], True, 'w'
    for i, tweet in enumerate(scraper.get_items()):
        tweets.append(tweet)
        if i % 1000 == 999:
            to_df(tweets, date).to_csv(fname, index=False, header=header, mode=mode, compression='gzip')
            tweets, header, mode = [], False, 'a'
    to_df(tweets, date).to_csv(fname, index=False, header=header, mode=mode, compression='gzip')
    final = pd.Timestamp.now()
    print(f'Date: {date.strftime("%Y-%m-%d")} | End: {final.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Date: {date.strftime("%Y-%m-%d")} | Dur: {(final - start).total_seconds()}s')

def to_df(tweets, date):
    # ['id', 'conversation_id', 'date', 'user_id', 'username', 'name',
    #  'tweet', 'replies_count', 'retweets_count', 'likes_count', 'link']
    df_dict = collections.OrderedDict([[k, []] for k in \
        ['id', 'conversationId', 'date', 'user_id', 'username', 'name',
         'content', 'replyCount', 'retweetCount', 'likeCount', 'url']])
    since = datetime.datetime(date.year, date.month, date.day, tzinfo=pytz.timezone('US/Pacific'))
    until = since + datetime.timedelta(days=1)
    for tweet in tweets:
        if since < tweet.date < until and tweet.lang == 'en':
            d = json.loads(tweet.json())
            for k in df_dict.keys():
                if k == 'user_id':
                    df_dict[k].append(d['user']['id'])
                elif k == 'username':
                    df_dict[k].append(d['user']['username'])
                elif k == 'name':
                    df_dict[k].append(d['user']['displayname'])
                else:
                    df_dict[k].append(d[k])
    df = pd.DataFrame.from_dict(df_dict)
    date = pd.DatetimeIndex(df['date'])
    if date.tzinfo is None:
        date = date.tz_localize('UTC')
    df['date'] = date.tz_convert(since.tzinfo)
    df.columns = ['id', 'conversation_id', 'date', 'user_id', 'username', 'name',
                  'tweet', 'replies_count', 'retweets_count', 'likes_count', 'link']
    return df

if __name__ == '__main__':
    processes = 7
    threads = 1
    dates = [
        '2015-07-13', '2015-07-14', '2015-08-11', '2015-08-12', '2015-08-13',
    ]

    size = math.ceil(len(dates) / processes)
    date_chunks = [dates[i:i+size] for i in range(0, len(dates), size)]
    query_chunks = [['#bitcoin']*len(c) for c in date_chunks]
    args = [(query, date, threads) for query, date in zip(query_chunks, date_chunks)]
    with mp.Pool(processes=processes) as pool:
        results = pool.starmap(scrape_multithread, args)
