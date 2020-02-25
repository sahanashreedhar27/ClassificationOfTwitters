"""
Wrapper for Twitter API.
"""
import os
from itertools import cycle
import sys
import time
import pandas as pd  # Data manipulation
import re, json
import tweepy as tp  # API to interact with twitter
from collections import defaultdict

RATE_LIMIT_CODES = set([88, 130, 420, 429])


class Twitter:
    def __init__(self, credential_file, directory=''):
        """
            Params: credential_file...list of JSON objects containing the four
            required tokens: consumer_key, consumer_secret, access_token, access_secret
		"""
        self.credentials = [json.loads(l) for l in open(credential_file)]
        self.credential_cycler = cycle(self.credentials)
        self.reinit_api()
        self.directory = directory

    def reinit_api(self):
        # creds = next(self.credential_cycler)
        auth = self.authenticate_twitter_app()
        self.twapi = tp.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

    def authenticate_twitter_app(self):
        creds = next(self.credential_cycler)
        sys.stderr.write('switching creds to %s\n' % creds['consumer_key'])

        # Authentication
        consumer_key = creds['consumer_key']
        consumer_secret = creds['consumer_secret']
        auth = tp.OAuthHandler(consumer_key, consumer_secret)

        # token stuff
        access_token = creds['access_token']
        access_token_secret = creds['token_secret']
        auth.set_access_token(access_token, access_token_secret)
        return (auth)

    def get_user_timeline_tweets(self, twitter_client, user_list, num_tweets):
        '''
        Uses Tweepy's cursor method to fetch a user's timeline tweets.
        :param twitter_client: the twitter client to use.
        :param user_list: List of users to collect data from.
        :param num_tweets: Number of tweets from user to collect
        :return: List of tweets
        '''
        tweets = []
        for user in user_list:
            print(f'Getting {num_tweets} tweets for {user}. ', end='')
            try:
                for tweet in tp.Cursor(twitter_client.user_timeline, id=user).items(num_tweets):
                    tweets.append(tweet)
            except tp.RateLimitError:
                print(f'SLEEPING DUE TO RATE LIMIT ERROR!!!!')
                time.sleep(15 * 60)
            except Exception as e:
                print(f'SOME ERROR OCCURRED...PASSING!!!')
                print(e.__doc__)
                pass
        return (tweets)

    # Helper function to get fixed number of tweets and put in results
    def get_tweets(self, twitter_client, v_user_list, num_tweets):
        '''
        Fetch tweets from user timeline and return dataframes
        :param twitter_client: the twitter client to use.
        :param v_user_list: List of users to collect data from.
        :param num_tweets: Number of tweets from user to collect
        :return: Dictonary of tweets and users dataframes
        '''
        statuses = self.get_user_timeline_tweets(twitter_client, v_user_list, num_tweets)

        # Create list to write to json file
        tweet_LoD, user_LoD = self.produce_status_LoDs(statuses)

        tweet_df = pd.DataFrame(tweet_LoD)
        user_df = pd.DataFrame(user_LoD)

        return (tweet_df, user_df)

    def produce_status_LoDs(self, statuses):
        '''
        Read in lists of statuses. Organize them.

        Args:
            list - statuses

        Returns:
            Cleaned dataframe, with an extra column: 'known_bot' = False
        '''
        tweet_LoD = []
        user_LoD = []
        for status in statuses:

            tweet_dict = {}
            user_dict = {}

            tweet_dict['user_id'] = status.author.id
            tweet_dict['user_screen_name'] = status.author.screen_name
            tweet_dict['created_at'] = str(status.created_at)
            tweet_dict['id'] = status.id
            tweet_dict['id_str'] = status.id_str
            tweet_dict['text'] = status.text
            tweet_dict['source'] = status.source
            tweet_dict['truncated'] = status.truncated
            tweet_dict['retweet_count'] = status.retweet_count
            tweet_dict['favorite_count'] = status.favorite_count
            tweet_dict['lang'] = status.lang
            tweet_dict['is_tweet'] = ((re.search('RT', status.text) == None))

            tweet_LoD.append(tweet_dict)

            # user data
            user_dict['name'] = status.author.name
            user_dict['screen_name'] = status.author.screen_name
            user_dict['description'] = status.author.description
            user_dict['followers_count'] = status.author.followers_count
            user_dict['location'] = status.author.location
            user_dict['friends_count'] = status.author.friends_count
            user_dict['listed_count'] = status.author.listed_count
            user_dict['favourites_count'] = status.author.favourites_count
            user_dict['statuses_count'] = status.author.statuses_count
            user_dict['has_bio'] = bool(user_dict['description'] not in ['NULL', 'NaN', '', ' ', pd.np.nan])
            user_dict['followers_count_gr_30'] = bool(user_dict['followers_count'] >= 30)
            user_dict['followers_2_times_ge_friends'] = bool(
                2 * user_dict['followers_count'] >= user_dict['friends_count'])
            user_dict['bot_in_biography'] = bool(
                type(user_dict['description']) is str and 'bot' in user_dict['description'].lower())
            user_dict['ratio_friends_followers_around_100'] = bool(
                user_dict['followers_count'] > 0 and 80.0 <= float(user_dict['friends_count']) / user_dict[
                    'followers_count'] >= 120.0)
            user_dict['no_location'] = bool(user_dict['location'] in ['NULL', 'NaN', '', ' ', pd.np.nan])

            if status.author.verified:
                user_dict['known_bot'] = False
            else:
                user_dict['known_bot'] = False
            if not any(d.get('id', None) == status.author.id for d in user_LoD):
                user_LoD.append(user_dict)
        print(f'number of tweets: {len(tweet_LoD)} and number of users: {len(user_LoD)} collected.')
        return (tweet_LoD, user_LoD)

    def produce_bot_LoDs(self, bots, for_bots=False):
        '''
        Read in dataframe of bots. Organize them.

        Args:
            dataframes - Bots

        Returns:
            Cleaned dataframe, with an extra column: 'known_bot' = True
        '''

        user_colnames = ['name', 'screen_name', 'description', 'location',
                         'friends_count', 'favourites_count', 'followers_count', 'listed_count',
                         'statuses_count', 'has_bio', 'followers_count_gr_30', 'bot_in_biography',
                         'no_location', 'followers_2_times_ge_friends', 'ratio_friends_followers_around_100',
                         'known_bot']

        bots['has_bio'] = bots['description'].apply(
            lambda x: False if str(x) not in ['NULL', 'NaN', '', ' ', pd.np.nan] else True)
        bots['followers_count_gr_30'] = bots['followers_count'].apply(lambda x: False if int(x) >= 30 else True)
        bots['bot_in_biography'] = bots['description'].apply(
            lambda x: False if type(x) is str and 'bot' in x.lower() else True)
        bots['no_location'] = bots['location'].apply(
            lambda x: False if x in ['NULL', 'NaN', '', ' ', pd.np.nan] else True)
        bots['followers_2_times_ge_friends'] = bots.apply(
            lambda row: True if 2 * row['followers_count'] >= row['friends_count'] else False, axis=1)
        bots['ratio_friends_followers_around_100'] = bots.apply(lambda row: True if row['followers_count'] > 0 and
                                                                                    80.0 <= float(
            row['friends_count']) / row['followers_count'] >= 120.0 else False, axis=1)

        bots['known_bot'] = True
        bots_output = bots[user_colnames]

        if for_bots:
            return (bots_output)
        else:
            return (bots_output.head(740))

    def fetch_bot_dataset_and_store(self):
        """
        Gathers genuine user data and processes from the local dataset located at project-samaritan1011001/osna/data/social_spambots_1.csv/users.csv
        :return: Nothing. Creates a file called b_user_table_out.json
        """
        bots = pd.read_csv("project-samaritan1011001/osna/data/social_spambots_1.csv/users.csv")

        bots_Clean = self.produce_bot_LoDs(bots, for_bots=True)
        print(f'Number of bots collected {len(bots_Clean)}')

        user_json = bots_Clean.to_json(orient='records')

        with open(self.directory + os.path.sep + 'b_user_table_out.json', 'w') as outfile:
            json.dump(user_json, outfile)

    def fetch_genuine_dataset_and_store(self):
        """
            Gathers genuine user data and processes from the local dataset located at project-samaritan1011001/osna/data/genuine_accounts.csv/users.csv
            :return: Nothing. Creates a file called g_user_table_out.json
        """
        g_users = pd.read_csv("project-samaritan1011001/osna/data/genuine_accounts.csv/users.csv")

        g_users_Clean = self.produce_bot_LoDs(g_users)
        print(f'Number of g users {len(g_users_Clean)}')

        user_json = g_users_Clean.to_json(orient='records')

        with open(self.directory + os.path.sep + 'g_user_table_out.json', 'w') as outfile:
            json.dump(user_json, outfile)

    def fetch_v_user_and_store(self, v_user_list, num_tweets):
        """
           Collects verified user data from the Twitter API and processes it
           :return: Nothing. Creates a file called v_user_table_out.json
        """
        # Get verified users, write them to HD
        v_tweet_df, v_user_df = self.get_tweets(self.twapi, v_user_list, num_tweets)
        print("Number of verified users COLLECTED: {}".format(len(v_user_df)))
        user_json = v_user_df.to_json(orient='records')
        tweet_json = v_tweet_df.to_json(orient='records')

        with open(self.directory + os.path.sep + 'v_tweet_table_out.json', 'w') as outfile:
            json.dump(tweet_json, outfile)
        with open(self.directory + os.path.sep + 'v_user_table_out.json', 'w') as outfile:
            json.dump(user_json, outfile)

    def fetch_nv_user_and_store(self, nv_user_list, num_tweets):
        """
           Collects unverified user data from the Twitter API and processes it
           :return: Nothing. Creates a file called nv_user_table_out.json
        """
        # Get unverified users, write them to HD
        nv_tweet_df, nv_user_df = self.get_tweets(self.twapi, nv_user_list, num_tweets)
        print("Number of unverified users COLLECTED: {}".format(len(nv_user_df)))
        user_json = nv_user_df.to_json(orient='records')
        tweet_json = nv_tweet_df.to_json(orient='records')

        with open(self.directory + os.path.sep + 'nv_tweet_table_out.json', 'w') as outfile:
            json.dump(tweet_json, outfile)
        with open(self.directory + os.path.sep + 'nv_user_table_out.json', 'w') as outfile:
            json.dump(user_json, outfile)

    def merge_bot_user_datasets(self):
        """
           Merges all the collected data into one json
           :return: Nothing. Creates a file called final_user_master.json
        """
        with open(self.directory + os.path.sep + 'v_tweet_table_out.json') as json_file:
            v_tweet_json = json.load(json_file)

        with open(self.directory + os.path.sep + 'v_user_table_out.json') as json_file:
            v_user_json = json.load(json_file)

        with open(self.directory + os.path.sep + 'nv_tweet_table_out.json') as json_file:
            nv_tweet_json = json.load(json_file)

        with open(self.directory + os.path.sep + 'nv_user_table_out.json') as json_file:
            nv_user_json = json.load(json_file)

        with open(self.directory + os.path.sep + 'g_user_table_out.json') as json_file:
            g_user_json = json.load(json_file)

        v_tweet_df = pd.read_json(v_tweet_json)
        v_user_df = pd.read_json(v_user_json)

        nv_tweet_df = pd.read_json(nv_tweet_json)
        nv_user_df = pd.read_json(nv_user_json)

        g_user_df = pd.read_json(g_user_json)

        # Merging v_users and nv_users
        user_df = nv_user_df.append(v_user_df, sort=False)  # , ignore_index=True)
        tweet_df = nv_tweet_df.append(v_tweet_df)  # , ignore_index=True)

        with open(self.directory + os.path.sep + 'b_user_table_out.json') as json_file:
            user_json = json.load(json_file)

        bots_Clean = pd.read_json(user_json)

        final_user_df = user_df.append(bots_Clean, sort=False)

        final_user_df = final_user_df.append(g_user_df, sort=False)
        print(f'TOTAL NUMBER OF USERS COLLECTED -> {len(final_user_df)}')

        with open(self.directory + os.path.sep + 'final_user_master.json', 'w') as outfile:
            json.dump(final_user_df.to_json(orient='records'), outfile)

    def findCommonNeighbors(self, influential_users):
        '''
        Finds common neighbors for a given list of influential users
        :param influential_users: List of users
        :return: List of common neighbors and a dictionary used to make the graph
        '''
        user_followers_dict = defaultdict(list)
        for inf_user in influential_users:
            try:
                for item in tp.Cursor(self.twapi.followers_ids, id=inf_user).items(limit=750):
                    user_followers_dict[inf_user].append(item)
            except tp.TweepError:
                print("tweepy.TweepError=", tp.TweepError)  # tweepy.TweepError
            except:
                e = sys.exc_info()[0]
                print("Error: %s" % e)  # print "error."

        ll_neighbors = [x for x in user_followers_dict.values()]

        result = set(ll_neighbors[0]).intersection(*ll_neighbors)
        final_nodes = list(result)
        edges_list_dict = []
        labels = {}
        for in_user in influential_users:
            final_nodes.append(in_user)
            labels[in_user] = in_user
            for i, y in enumerate(list(result)):
                edges_list_dict.append((in_user, y))
        return list(result), edges_list_dict

    def fetch_test_data(self, test_list, num_tweets):
        '''
        Fetchs data for the given test user list
        :param test_list: List of users.
        :param num_tweets: Number of tweets to fetch for each user.
        :return: A josn object with the users' data.
        '''
        test_df, test_user_df = self.get_tweets(self.twapi, test_list, num_tweets)
        tweet_json = test_df.to_json(orient='records')
        user_json = test_user_df.to_json(orient='records')

        return user_json

    # EXTRA FUNCTIONS : IGNORE

    # def user_info_for_screen_name(self, screen_name):
    #     response = self.twapi.get_user(screen_name)
    #     print(f' user info -> {response.followers_count}')
    #     return response.followers_count
    #
    # def followers_count_for_users(self, users_list):
    #     reach_count = 0
    #     users = self.twapi.lookup_users(users_list)
    #     for user in users:
    #         reach_count += user.followers_count
    #     return reach_count
    #
    # def get_all_statuses(self, screen_name):
    #     statuses = []
    #     for status in tp.Cursor(self.twapi.user_timeline, screen_name=screen_name, tweet_mode="extended").items():
    #         statuses.append(status)
    #     print(f'Total number of statuses retreived -> {len(statuses)}')
    #     # print(f'A status -> {[x.id for x in statuses if x.retweet_count > 0]}')
    #     statuses_rt_ge_0 = [x.id for x in statuses if x.retweet_count > 0]
    #     print(f'Total number of statuses_rt_ge_0 retreived -> {len(statuses_rt_ge_0)}')
    #
    #     statuses_rt_ge_0 = statuses_rt_ge_0[:75]
    #     return self.get_retweeters_id_for_statuses(statuses_rt_ge_0)
    #
    # def get_retweeters_id_for_statuses(self, statuses):
    #     retweeters_ids = []
    #     reach_count = 0
    #     for status_id in statuses:
    #         for retweeters in tp.Cursor(self.twapi.retweeters, id=status_id, tweet_mode="extended").pages():
    #             print(f'Type -> {type(retweeters)}')
    #             retweeters_ids.append(retweeters)
    #             reach_count += self.followers_count_for_users(retweeters)
    #             # self.user_info_for_screen_name()
    #     print(f'Total number of retweeters retreived -> {retweeters_ids[0]}')
    #     print(f'reach_count -> {reach_count}')
    #     return reach_count
