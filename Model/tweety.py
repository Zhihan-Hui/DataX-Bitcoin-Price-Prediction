
import multiprocessing as mp
import pandas as pd
import numpy as np
import collections
import threading
import calendar
import asyncio
import queue
import twint
import math
import time
import pytz
import uuid
import sys
import os
import re

# USAGE NOTES:
#
#   - IMPORTANT: it is HIGHLY recommended that the number of processes times the number of threads is 
#     equal to the number of days to process or slightly greater than the number of days to process.
#     In other words, there should be a sufficient amount of workers for the number of days you want to 
#     process. This is automatically done if the -t flag is omitted. Also, making the number of processes 
#     greater than the number of threads (while still making sure there are enough workers) can lead to even
#     better performance. However, this may start to slow down other processes currently working on your
#     computer. If this is not desired, then using more threads is the way to go. For whatever reason, twint 
#     will hang for quite a while after processing tweets for a given range. This means that if there are less
#     workers than the number of days you want to process you'll need to wait for a pretty long time (see 
#     timing notes). To cut down on this wait time, it's better to process all tweets in parallel in one run.
#
#   - If the program detects that it has already processed a certain date and query from a previous run, it will
#     NOT process that date and query again. Instead, it will simply leave the pre-computed file alone. The way
#     the program detects past work is through file names so please try not to rename the files it has generated.
#     Instead make copies of the files and move them to your own directory for renaming.
#
#   - The program will finish collecting tweets for some days faster than others. The files that end with the
#     string '-compressed.csv' (and aren't changing in size) have been fully processed and may be copied to
#     another folder for further use.
#
#   - For some reason, when twint finishes collecting tweets for a certain day, it will hang for 5 - 10 minutes.
#     Although it may seem like the program is frozen during this time, it is actually still working. Please be 
#     patient and do not close the terminal unless an unreasonable amount of time has passed (20 - 30 minutes)
#     without something happening.
#
#   - To view the progress of the program, you can check the sizes of each file or see which files have
#     '-compressed.csv' in their name.
#
#   - The program will NOT stop gracefully if you try to exit the terminal or press keyboard interrupt. In other 
#     words, if you try to stop the program early, some processes and/or threads won't get shut down properly and
#     may even continue working. If this is the case, you'll have to use the task manager to shut them down.
#
# TIMING NOTES:
#
#   - The time it takes for the program to run depends on the time it takes to process the day with the largest amount of tweets.
#     With that in mind, it is recommended to keep date ranges relatively small (<= 31 days) to reduce the chances of coming 
#     across a date with a large number of tweets.
#
#   - Timing statistics are given below. For each task, 31 days worth of tweets (2017-12-15 to 2018-01-15 (exclusive)) were collected:
#       
#       - BASELINE:
#           - processes: 1
#           - threads per process: 31
#           - GOOD: 1 * 31 = 31 >= 31 (there will be one thread per day)
#           - total time taken: 1 hour(s), 34 minute(s), and 44 second(s)
#
#       - WORKERS < JOBS (NOT RECOMMENDED):
#           - This is around 18 minutes less than the baseline time!
#           - processes: 5
#           - threads per process: 5
#           - BAD: 5 * 5  = 25 < 31 (not enough threads for each day)
#           - total time taken: 1 hour(s), 16 minute(s), and 11 second(s) 
#
#       - WORKERS >= JOBS (RECOMMENDED IF YOU DON'T WANT TO SLOW DOWN OTHER PROCESSES ON YOUR COMPUTER):
#           - This is around 44 minutes less than the baseline time!
#           - processes: 5
#           - threads per process: 7 thread(s) 
#           - GOOD: 5 * 7 = 35 >= 31 (there will be one thread per day)
#           - total time taken: 0 hour(s), 50 minute(s), and 06 second(s)
#
#       - WORKERS >= JOBS (RECOMMENDED IF YOU DON'T MIND SLOWING DOWN OTHER PROCESSES ON YOUR COMPUTER):
#           - This is around 54 minutes less than the baseline time!
#           - processes: 7
#           - threads per process: 5 thread(s) 
#           - GOOD: 7 * 5 = 35 >= 31 (there will be one thread per day)
#           - total time taken: 0 hour(s), 40 minute(s), and 4 second(s)
#
#   - In an older version of this program, collecting ALL tweets in 2018 took around 4 hours on my machine.
#       - 12 processes were used
#       - Each process handled one month

OUTPUT_FLDR_PRFX = 'tweets'
OUTPUT_FILE_NAME = 'tweets'

# https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
# https://stackoverflow.com/questions/17557074/memory-error-when-using-pandas-read-csv
DTYPES = {
    'cashtags' : object,
    'conversation_id' : str,
    'created_at' : str,
    'date' : str,
    'geo' : object,
    'hashtags' : object,
    'id' : str,
    'language' : str,
    'likes_count' : str,
    'link' : str,
    'mentions' : object,
    'name' :str,
    'near' : object,
    'photos' : object,
    'place' : object,
    'quote_url' : str,
    'replies_count' : str,
    'reply_to' : object,
    'retweet' : object,
    'retweet_date' : str,
    'retweet_id' : str,
    'retweets_count' : int,
    'source' : object,
    'thumbnail' : str,
    'time' : str,
    'timezone' : str,
    'trans_dest' : object,
    'trans_src' : object,
    'translate' : object,
    'tweet' : str,
    'urls' : object,
    'user_id' : str,
    'user_rt' : object,
    'user_rt_id' : str,
    'username' : str,
    'video' : str
}

def extract_year(date_str):
    return date_str[:date_str.find('-')]

def extract_month(date_str):
    return date_str[date_str.find('-')+1 : date_str.rfind('-')]

def get_path(prefix, date_str, suffix=''):
    return os.path.join(
        OUTPUT_FLDR_PRFX + prefix, extract_year(date_str), extract_month(date_str), 
        f'{date_str}-{OUTPUT_FILE_NAME}{suffix.replace(" ", "-")}.csv'
    )

def get_config(query, since, until):
    config = twint.Config()
    config.Search = query
    config.Lang = 'en'
    config.Since = since
    config.Until = until
    config.Hide_output = True
    config.Store_csv = True
    config.Output = get_path(query, since, suffix='-'+config.Search)

    # For some reason, twint changes Since and Until under the
    # hood, so, we'll put the real values in our own variables
    config.since = since
    config.until = until

    return config

def get_inputs(start, end, timezone, query, processes, threads):
    # NOTE: To get all tweets on say 2018-01-01, you need to specify that:
    # since = 2018-01-01 AND until = 2018-01-03 (i.e. you need 2 extra days to avoid data loss)
    chunk = math.ceil((end - start).days / processes)
    confs = []
    for d in pd.date_range(start, end, closed='left'):
        since = d.strftime('%Y-%m-%d')
        until = (d + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        confs.append(get_config(query, since, until))   
    return [(confs[i:chunk+i], timezone, threads) for i in range(0, len(confs), chunk)]

def process_chunk(chunk, since, o_path, **to_csv_kwargs):
    copy = chunk.copy()
    copy['date'] = pd.DatetimeIndex(copy['created_at']).tz_convert(since.tz)
    copy.drop(columns=['created_at', 'time', 'timezone', 'place', 'language',
                       'mentions', 'urls', 'photos', 'hashtags', 'cashtags',
                       'retweet', 'quote_url', 'video', 'thumbnail', 'near', 'geo',
                       'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                       'retweet_date', 'translate', 'trans_src', 'trans_dest'], inplace=True)
    copy = copy[(copy['date'] >= since.replace(hour=00, minute=00, second=00)) & \
                (copy['date'] <= since.replace(hour=23, minute=59, second=59))]
    copy.to_csv(o_path, **to_csv_kwargs)

def task(q, timezone, stop_event):
    asyncio.set_event_loop(asyncio.new_event_loop())
    while not stop_event.is_set():
        try:
            config = q.get(timeout=0.5)
            i_path = get_path(config.Search, config.since, suffix='-'+config.Search)
            o_path = get_path(config.Search, config.since, suffix=f'-{config.Search}-compressed')
            if not os.path.exists(o_path):
                twint.run.Search(config)
                since = pd.Timestamp(config.since, tz=timezone)
                readr = pd.read_csv(i_path, dtype=DTYPES, chunksize=1000)
                try:
                    process_chunk(next(readr), since, o_path
                        , index=False
                        , header=True
                        , mode='w'
                        , compression='gzip'
                    )
                    for chunk in readr:
                        process_chunk(chunk, since, o_path
                            , index=False
                            , header=False
                            , mode='a'
                            , compression='gzip'
                        )
                except StopIteration as e:
                    pass
                os.remove(i_path)
        except queue.Empty:
            continue
        q.task_done()

def scrape(configs, timezone, max_threads):
    thread_lst = []
    stop_event = threading.Event()
    outpt_lock = threading.Lock()
    work_queue = queue.Queue()

    for c in configs:
        work_queue.put(c)

    for i in range(max_threads):
        thread_lst.append(
            threading.Thread(
                target=task, 
                args=(work_queue, timezone, stop_event)
            )
        )
        thread_lst[i].start()

    work_queue.join()
    stop_event.set()
    for t in thread_lst: t.join()

def create_folders(prefix, start, end):
    directory = OUTPUT_FLDR_PRFX + prefix
    for d in pd.date_range(start, end, closed='left'):
        year = os.path.join(directory, str(d.year))
        mnth = extract_month(d.strftime('%Y-%m-%d'))
        path = os.path.join(year, mnth)
        if not os.path.exists(directory):
            os.mkdir(directory)
        if not os.path.exists(year):
            os.mkdir(year)
        if not os.path.exists(path):
            os.mkdir(path)
    
def take_input_if(cond, msg=''):
    if cond:
        are_you_sure = ''
        msg += " Would you like to continue?\n(y/n): "
        while not re.match(r'^[y|n]$', are_you_sure.lower()):
            are_you_sure = input(msg)
        if are_you_sure == 'n':
            sys.exit()

class TweetScraper:
    """
    A wrapper class for parallel processing tweets using the twint module.
    """

    _tweet_data = None
    _twtr_query = None
    _tweet_lock = None
    _stop_event = None
    _is_updated = None
    _twtrlogger = None
    _twtrwriter = None
    _prev_write = None  
    _next_write = None

    def __init__(self, realtime_mode=False, query=None):
        """
        There are two ways to use this class:

            1. For obtaining historical tweet data (set `realtime_mode` to False)

            2. For real time data (set `realtime_mode` to True)
    
        Option 1 is recommended if you expect to spend more time analyzing past trends.
        Option 2 is recommended if you want to use live tweets. If you use option 2, then when
        the first instance of this class is created, a cache will be filled with all tweets 
        from the past 24 hours. The cache is shared between ALL instances of this class to prevent
        waste. This means that updates to the cache will be seen by all instances. Similarly, 
        turning off caching from one instance will cause it to stop for all instances.
        """
        self.query = query
        if realtime_mode and TweetScraper._twtrlogger is None:
            TweetScraper._twtr_query = query
            TweetScraper._tweet_data = collections.deque()
            TweetScraper._tweet_lock = threading.Lock()
            TweetScraper._stop_event = threading.Event()
            TweetScraper._is_updated = threading.Event()
            TweetScraper._twtrlogger = threading.Thread(
                target=self._logdata,
                args=()
            )
            TweetScraper._twtrlogger.start()

    def _dfcache(self, wait):
        if wait: TweetScraper._is_updated.wait()
        return pd.concat(list(TweetScraper._tweet_data), ignore_index=True)\
                 .drop_duplicates(subset='id', keep='last')\
                 .sort_values('date', ascending=False)\
                 .reset_index(drop=True)

    def _writedata(self, snapshot):
        start = snapshot['date'].min()
        final = start + pd.Timedelta(days=1)
        create_folders(self.query, start, final)
        path = get_path(self.query, start.strftime('%Y-%m-%d'), suffix=f'-{TweetScraper._twtr_query}-compressed')
        snapshot.to_csv(path
            , index=False
            , header=True
            , mode='w'
            , compression='gzip'
        )

    def _last_log(self):
        if len(TweetScraper._tweet_data) == 0:
            return (pd.Timestamp.now() - pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
        return TweetScraper._tweet_data[-1]['date'].max().strftime('%Y-%m-%d %H:%M:%S')

    def _warm_up(self, workspace='tweet_cache'):
        # Clear any existing files
        if os.path.exists(workspace):
            for f in os.listdir(workspace):
                os.remove(os.path.join(workspace, f))

        # The inner call to `scrape_recent(...)` takes a while because the cache is cold.
        # The call to `refresh(...)` collects all tweets missed in the time it took to 
        # warm up the cache. This makes it possible to use `scrape_by_timeout` efficiently.
        one_day_ago = pd.Timestamp.now() - pd.Timedelta(days=1)
        one_day_ago = one_day_ago.strftime('%Y-%m-%d %H:%M:%S')
        TweetScraper._tweet_data.append(
            self.refresh(
                self.scrape_recent(one_day_ago
                    , query=TweetScraper._twtr_query
                    , dirname=workspace
                    , sleep=2
                    , timeout=float('inf')
                )
                , query=TweetScraper._twtr_query
                , dirname=workspace
                , sleep=2
                , timeout=float('inf')
            )
        )

    # Future work:
    #   - The code in `_warm_up` was written with the assumption that `refresh` doesn't take too long. 
    #     This assumption is valid when the query is bitcoin, but it may not be true for other queries.
    #   - Make it so that TweetScraper._tweet_data[i] is a dataframe of tweets within the same minute.
    #     This will make it easier to design a cache refreshment mechanism.
    def _logdata(self):
        self._warm_up()
        time.sleep(60 - pd.Timestamp.now().second)
        while not TweetScraper._stop_event.is_set():
            TweetScraper._tweet_lock.acquire()

            # We start from the last recorded date, so there will
            # be some duplicates here.
            new_tweets = self.scrape_by_timeout(self._last_log()
                , TweetScraper._twtr_query
                , timeout=3
                , dirname='tweet_cache'
            )

            if len(new_tweets) > 0:
                TweetScraper._tweet_data.append(new_tweets)

            TweetScraper._tweet_lock.release()
            TweetScraper._is_updated.set()
            time.sleep(55 - pd.Timestamp.now().second)
            TweetScraper._is_updated.clear()
            time.sleep(60 - pd.Timestamp.now().second)

    def get_recent(self):
        """
        Get the most recent tweets stored in the cache. If the cache is currently updating,
        this function will wait for the update to complete and return the tweets collected
        in that update.

        Returns:
        --------
            A pandas dataframe containing the most recent tweets collected by the cache. If
            caching is off, this function returns None. 
        """     
        if self.is_caching():
            TweetScraper._is_updated.wait()
            return TweetScraper._tweet_data[-1]

    def get_cache(self):
        """
        Get all data currently in the cache. If the cache is currently being updated, this 
        function will wait for the update to complete before retrieving the cache.

        Returns:
        --------
            A pandas dataframe containing the contents of the cache. If caching is off, this 
            function returns None.
        """
        if self.is_caching(): return self._dfcache(wait=True)

    def is_updated(self):
        """
        Returns True if the cache has data for the curent minute and False 
        otherwise.

        Notes:
        ------
            If caching is off, this function returns None.
        """
        if self.is_caching(): return TweetScraper._is_updated.is_set()

    def is_caching(self):
        """
        Returns True if caching is active and False otherwise.
        """
        return TweetScraper._stop_event is not None and not TweetScraper._stop_event.is_set()

    def end_caching(self, wait=True):
        """
        Turns tweet caching off. Only has an effect if realtime mode is on.
        If there are multiple instances of this class, turns off price caching
        for ALL of them.

        Parameter(s):
        -------------
            wait : bool
                If True, wait for the thread to fully finish running. If
                False, return control immediately and let the thread exit
                on its own.
        """
        if self.is_caching():
            TweetScraper._stop_event.set()
            if wait: TweetScraper._twtrlogger.join()

    def collect_tweets_on(self, date, query):
        """
        Get a dataframe of tweets on day `date` using the compressed CSV files 
        generated by the program. If no file exists for a given day, then this 
        function will return None.

        Parameter(s):
        -------------
            start : string
                A string of the form YYYY-mm-dd representing the day to collect tweets for.

            query : string
                A string representing the search term used to generate the file.

        Returns:
        --------
            A dataframe of tweets for the specified time interval and query.
        """
        path = self.path_to(date, query)
        if os.path.exists(path):
            return pd.read_csv(path
                , dtype=DTYPES
                , compression='gzip'                    
            )

    def collect_tweets_between(self, start, final, query):
        """
        Get a dataframe of all tweets from `start` to `final` using the 
        compressed CSV files generated by the program. If no file exists
        for a given day, then this function will simply move onto the next
        day. If the time range is too large to fit in a dataframe, an error
        will be raised.

        Parameter(s):
        -------------
            start : string
                A string of the form YYYY-mm-dd representing the first day (inclusive) 
                to start at.

            final : string
                A string of the form YYYY-mm-dd representing the last day (exclusive) 
                of the time interval.

            query : string
                A string representing the search term used to generate the file.

        Returns:
        --------
            A dataframe of tweets for the specified time interval and query.
        """
        
        return pd.concat([self.collect_tweets_on(d.strftime('%Y-%m-%d %H:%M:%S'), query) \
            for d in pd.date_range(start, final, closed='left')], ignore_index=True)

    def get_dtypes(self):
        """
        Get a dictionary from column names to data types for the dtype 
        parameter of `pandas.read_csv(...)`.

        Notes:
        ------
            The columns in the twitter data that should be integers were converted
            to strings to avoid NaN errors. This means that after you use `read_csv`
            you'll need to convert the numerical columns from string to their 
            appropriate type.
        """
        return DTYPES.copy()

    def path_to(self, date, query):
        """
        A helper method for getting the path to one of the files generated by 
        the program. Files are uniquely identified by date, query, and suffix. 
        This method does not check if the file exists, it simply returns the 
        path to the file as if it exists.

        Parameter(s):
        -------------
            date : string
                A string of the form YYYY-mm-dd representing the day that tweets
                were collected on.

            query : string
                A string representing the search term used to generate the file.

        Returns:
        --------
            A string representing the path to one of the automatically-generated 
            files created by the program (this file may or may not exist).
        """
        return get_path(query, date, suffix=f'-{query}-compressed')

    def _getrecent(self, fname, since, query, close_file=True):
        asyncio.set_event_loop(asyncio.new_event_loop())
        config = twint.Config()
        config.Search = query
        config.Lang = 'en'
        config.Since = since
        config.Hide_output = True
        config.Store_csv = True
        config.Output = fname
        twint.run.Search(config)
        if close_file: os.remove(fname)

    def _checkfile(self, fname, q, since, query, sleep, timeout):
        """
        This function will check if the size of the CSV file created by `_getrecent(...)`
        has changed every `sleep` seconds. If no change in the size is detected (and the
        `timeout` hasn't expired), then the contents of the CSV file are stored in `q` as 
        a pandas DataFrame.

        Parameter(s):
        -------------
            q : queue.Queue()
                The queue to store the result in.

            sleep : int
                The number of seconds to wait in between checks.

            timeout : int
                The maximum number of seconds to spend looping.

        Returns:
        --------
            None. Stores a dataframe with the resulting data in the specified queue.
            If timeout expires before the file finishes updating, then None is stored
            in the queue. If the file generated by the program is not found within 
            the timeout, an exception is raised.
        """
        # Repeatedly search for the file first
        start = time.time()
        while not os.path.exists(fname):
            if time.time() - start >= timeout:
                raise Exception(f'Could not find file {fname}')
        time.sleep(sleep)

        # If the file exists, wait until its size stops changing
        prev_size = os.path.getsize(fname)
        time.sleep(sleep)
        start = time.time()
        success = False
        while not success and time.time() - start < timeout:
            curr_size = os.path.getsize(fname)
            if curr_size != prev_size:
                prev_size = curr_size
                time.sleep(sleep)
            else:
                success = True

        # If we waited long enough for the file to update completely,
        # store its dataframe representation inside the queue. Otherwise
        # store None in the queue.
        if success:
            df = pd.read_csv(fname, dtype=DTYPES)
            df['date'] = df['created_at']
            df.drop(columns=['created_at', 'time', 'timezone', 'place', 'language',
                             'mentions', 'urls', 'photos', 'hashtags', 'cashtags',
                             'retweet', 'quote_url', 'video', 'thumbnail', 'near', 'geo',
                             'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                             'retweet_date', 'translate', 'trans_src', 'trans_dest'], inplace=True)
            q.put(df)
        else:
            q.put(None)

    def scrape_recent(self, since, query, dirname=None, sleep=2, timeout=float('inf')):
        """
        Collect tweets up to 48 hours in the past. The closer `since` is 
        to the current time, the faster this function will run. This is 
        different from `scrape_by_day(...)` because the `since` argument
        may now include a time.

        Parameter(s):
        -------------
            since : string
                The date to start the scrape from. Must be in the format
                YYYY-mm-dd HH:MM:SS. Must be within 24 hours of the current
                time. For larger requests, use the other functions.

            query : string
                The search term.

            dirname : string
                The folder to place scratch CSV files.

            sleep : int
                The number of seconds to sleep between file checks. You 
                shouldn't need to adjust this, but it is made as a parameter
                just in case.

            timeout : float('inf') or int
                The maximum amount of time we should wait for the scrape.

        Returns:
        --------
            If the timeout expires, None is returned. Otherwise, returns
            a dataframe with the specified data.

        Notes:
        ------
            Since twint hangs for a while after processing a request, you may
            see the CSV file used by this function before it is fully removed.
            Feel free to ignore it.
        """
        # This makes sure we don't get any file naming errors
        fname  = str(uuid.uuid4()) + '.csv'
        if dirname is not None:
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            fname = os.path.join(dirname, fname)

        # Check that `since` is not too far in the past. If we don't restrict 
        # the date, then we may run into memory errors.
        since_dt = pd.Timestamp(since)
        if (pd.Timestamp.now() - since_dt).total_seconds() > (86400 * 2) + 1:
            raise ValueError("Time range to large.")

        # This will store the resulting data
        q = queue.Queue()

        # A thread that performs the scrape
        twint_thread = threading.Thread(
            target=self._getrecent,
            args=(fname, since, query, )
        )
        twint_thread.start()

        # A thread for getting the results of the scrape faster
        check_thread = threading.Thread(
            target=self._checkfile,
            args=(fname, q, since, query, sleep, timeout, )
        )
        check_thread.start()

        # Wait for the check thread to get the data
        check_thread.join()
        return q.get(timeout=0.5)

    def refresh(self, data_frame, query, offset=0, shrink=False, dirname=None, sleep=2, timeout=float('inf')):
        """
        Gathers any new tweets from the latest date in `data_frame` to now.

        Parameter(s):
        -------------
            data_frame : DataFrame
                A dataframe of twint's twitter data.

            query : string
                The search term used to generate this dataframe.

            shrink : bool
                If False (the default), returns only the new data points. 
                Otherwise, returns df concatenated with the new data points.

            offset : int
                If specified, refresh data starting from the latest date in
                `data_frame` minus `offset` minutes up to the current time. 
                If this new time range is too large (i.e. larger than the max 
                time range in `scrape_recent(...)`, an error is raised. By 
                default, we refresh starting from the most recent time in 
                `data_frame`.

            dirname : string or None
                The folder to place scratch CSV files. If None, places files in 
                the current working directory.

            sleep : int
                The number of seconds to sleep between file checks. You 
                shouldn't need to adjust this, but it is made as a parameter
                just in case the result is being returned too early.

            timeout : float('inf') or int
                The maximum amount of time we should wait for the scrape.

        Returns:
        --------
            A new dataframe with the original data plus any new data. If `shrink`
            is True, returns a dataframe with only the new data points.

        Notes:
        ------
            The time range for scraping must be less than one day or an error will be
            raised.

            This function assumes that `data_frame` is already sorted on "date" where 
            the most recent time is the very first row.
        """
        since = data_frame['date'].max()
        since = since - pd.Timedelta(minutes=offset)
        since = since.strftime('%Y-%m-%d %H:%M:%S')
        extra = self.scrape_recent(since, query, dirname=dirname, sleep=sleep, timeout=timeout)
        if extra is None: return None
        if shrink: return extra 
        data_frame = data_frame[data_frame['date'] < since]
        return pd.concat([extra, data_frame], ignore_index=True)

    def scrape_by_timeout(self, since, query, timeout=None, dirname=None):
        """
        Scrape all tweets related to `query` and use a timeout to
        retrieve results faster.

        Parameter(s):
        -------------
            since : string
                The date to start the scrape from. Must be in the format
                YYYY-mm-dd HH:MM:SS.
            
            query : string
                The search term.

            timeout : int or None
                The maximum time to spend waiting for the scrape.
                If None, wait for twint to run in its entirety. 
                If an int, collect the data as soon as the 
                specified number of seconds have passed.

            dirname : string or None
                The folder to place scratch CSV files. If None, places files in 
                the current working directory.

        Returns:
        --------
            A pandas dataframe with tweets from the past minute.

        Notes:
        ------
            It is recommended to first check how long twint waits to
            collect tweets then adjust the timeout accordingly. A CSV
            file with a random name will be generated in the current
            directory when this function is called. You can inspect 
            the time it takes for the file's size to stop changing 
            and use that as a rough guide to determine the timeout. 
        """
        # This makes sure we don't get any file naming errors
        fname  = str(uuid.uuid4()) + '.csv'
        if dirname is not None:
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            fname = os.path.join(dirname, fname)

        if timeout is None:
            self._getrecent(fname, since, query, False)
            df = pd.read_csv(fname)
            os.remove(fname)
        else:
            twint_thread = threading.Thread(
                target=self._getrecent,
                args=(fname, since, query, True, )
            )
            twint_thread.start()
            time.sleep(timeout)
            df = pd.read_csv(fname)
        df['date'] = pd['created_at']
        return df.drop(columns=['created_at', 'time', 'timezone', 'place', 'language',
                                'mentions', 'urls', 'photos', 'hashtags', 'cashtags',
                                'retweet', 'quote_url', 'video', 'thumbnail', 'near', 'geo',
                                'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                                'retweet_date', 'translate', 'trans_src', 'trans_dest'])

    def scrape_by_day(self, date, timezone, query, verbose=True):
        """
        Scrape all tweets related to `query` on a particular day.

        Parameter(s):
        -------------
            date : string
                A string of the form YYYY-mm-dd representing the day to scrape.

            timezone: pytz.timezone
                A pytz.timezone object specifying the timezone for date.

            query : string
                The search term.

            verbose : bool
                If True, prints some helper messages.

        Returns:
        --------
            None. Writes all data to a compressed CSV file.
        """
        start = pd.Timestamp(date)
        next_day = start + pd.Timedelta(days=1)
        self.scrape_by_range(date, next_day, timezone, query, processes=1, threads=1, verbose=verbose)

    def scrape_by_month(self, date, timezone, query, processes=None, threads=None, verbose=True):
        """
        Scrape all tweets related to `query` on a particular month.

        Parameter(s):
        -------------
            date : string
                A string of the form YYYY-mm representing the year and month 
                to scrape.

            timezone: pytz.timezone
                A pytz.timezone object specifying the timezone for date.

            query : string
                The search term.

            verbose : bool
                If True, prints some helper messages.

        Returns:
        --------
            None. Writes all data to compressed CSV files.
        """
        start = pd.Timestamp(date)
        final = start + pd.Timedelta(days=calendar.monthrange(start.year, start.month)[1])
        self.scrape_by_range(start, final, timezone, query, processes=processes, threads=threads, verbose=verbose)

    def scrape_by_range(self, start, final, timezone, query, processes=None, threads=None, verbose=True):
        """
        Scrape all tweets related to `query` for each day in between `start` 
        (inclusive) and `final` (exclusive).

        Parameter(s):
        -------------
            start : string or pd.Timestamp
                A string of the form YYYY-mm-dd or pd.Timestamp representing
                the first day (inclusive) to scrape.

            final : string or pd.Timestamp
                A string of the form YYYY-mm-dd or pd.Timestamp representing
                the last day (exclusive) to scrape.

            timezone: pytz.timezone
                A pytz.timezone object specifying the timezone for start and final.

            query : string
                The search term.

            processes : int
                The number of processes to spawn.

            threads : int
                The maximum number of threads for each process to use.

            verbose : bool
                If True, prints some helper messages.

        Returns:
        --------
            None. Writes all data to compressed CSV files.
        """
        start = pd.Timestamp(start)
        final = pd.Timestamp(final)
        today = pd.Timestamp.now()
        if start >  today: raise ValueError('Start date must not be in the future.')
        if start >= final: raise ValueError('Start date must come before end date.')
        if final >  today: final = final.replace(year=today.year, month=today.month, day=today.day+1)

        # START OF BASIC SAFETY CHECKS

        try:
            ndays = (final - start).days
            take_input_if(ndays > 365, 
                "You are about to process more than a years worth of data."
            )
            take_input_if(processes is None and threads is None, 
                "You are about to use all processing power available."
            )
            if processes    is None: processes  = mp.cpu_count()
            if threads      is None: threads    = math.ceil(ndays / processes)
            wrkrs = processes * threads
            take_input_if(ndays > wrkrs, 
                f"Number of workers ({wrkrs}) is less than the number of jobs ({ndays}). This may take a while."
            )
        except SystemExit as e:
            print("Exiting...")
            return

        # END OF BASIC SAFETY CHECKS

        if verbose: print("Starting the scraper!")
        create_folders(query, start, final)
        if verbose: print("Directories created!")

        inputs = get_inputs(start, final, timezone, query, processes, threads)

        start = pd.Timestamp.now()
        if verbose: 
            print('Start time: ', start.strftime('%Y-%m-%d %H:%M:%S'))
            print('Processing...')
            sys.stdout.flush()

        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(scrape, inputs)

        final = pd.Timestamp.now()
        if verbose: 
            print('Final time: ', final.strftime('%Y-%m-%d %H:%M:%S'))
            print('Total time: ', final - start)
            sys.stdout.flush()

    def _clean_txt(self, text):
        text = re.sub(r"(?:\@)\S+", "", text) #Removing @mentions
        text = re.sub('#', '', text) #Removing '#' hash tag
        text = re.sub('RT[\s]+', '', text) #Removing RT
        text = re.sub('https?:\/\/\S+', '', text) #Removing hyperlink
        text = text.strip() #Removing leading and trailing spaces 
        text = text.lower() #Make tweet text lowercase
        return text


if __name__ == '__main__':
    scraper = TweetScraper()
    timezone = pytz.timezone('US/Pacific')
    processes, threads = 8, 4

    query = '#bitcoin'
    dates = pd.date_range('2015-01', '2015-12', freq='MS')
    for date in dates:
        scraper.scrape_by_month(date, timezone, query, processes=processes, threads=threads)
