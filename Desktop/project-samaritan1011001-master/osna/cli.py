# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.



import click
import json
import pickle
import sys

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from . import credentials_path, clf_path
from .mytwitter import Twitter
import time
import pandas as pd
import matplotlib.pyplot as plt  # Plotting
import networkx as nx
import seaborn as sns
from tabulate import tabulate

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('collect')
@click.argument('directory', type=click.Path(exists=True))
def collect(directory):
    """
    Collect data and store in given directory.

    This should collect any data needed to train and evaluate your approach.
    This may be a long-running job (e.g., maybe you run this command and let it go for a week).
    """

    twitter = Twitter(credentials_path, directory)

    # Define list of unverified users.
    nv_user_list = ['@sairammurthy6', '@slactochile', '@jaymijams', '@ChefDoubleG',
                    '@ili44ili.', '@Rcontreras777', '@MissMaseline', '@mike434prof', '@NonativeEuan',
                    '@bodmonbrandon_', '@tastytasy', '@jamesplegg', '@esruben', '@ObVents', '@YusufLaReveur',
                    '@TurnbowRosemary', '@todaav', '@Pasho53013866', '@tonyaba18632641', '@ghostsignal1',
                    '@lxcleopatraxl', '@NazarethLGP', '@MockZui', '@onegearrico', '@abadreen',
                    '@somerice', '@unsaltCarthage', '@Cmiln01', '@Kitter_44', '@ashish3vedi',
                    '@HugoMunissa', '@TODthegiant', '@LissyBee4', '@anna_adamcova', '@jerwinbroas2',
                    '@MockZui', '@heavyoilcountr1', '@RajeshHitha', '@rjerome217', '@louisftelu',
                    '@antimickey_', '@guywpt', '@bernoroel', '@DavidOrr4', '@FarajShaikh6',
                    '@LegionHoops', '@wrwveit', '@TriggaGhj', '@duckmesick', '@tyjopow', '@mskoch',
                    '@jaspect_wan', '@WiseSparticus', '@Mr_AdiSingh', '@Live9Fortknox', '@mrfridberg',
                    '@vibolnet', '@paulanderson801', '@AmirRozali', '@sumitakale', '@MoonWorld__94',
                    '@itselijahgm', '@S_Nenov', '@HglundNiklas', '@LBoertjes', '@MulaMutha', '@iantuck99',
                    '@JumahSaisi', '@onlygetbright', '@iamPavanRayudu', '@LeeThecritch', '@mkinisa1',
                    '@Anfieldvianne', '@DonUbani', '@JardyRaines', '@BagbyCarole', '@JopiHuangShen',
                    '@scottwms84', '@gander99', '@biller_jon', '@JLeeAURivals', '@ramya', '@LambdaChiASU',
                    '@joey_gomez', '@anthoamick844', '@Brettwadeart', '@zac_slocumb', '@NatoNogo', '@Twu76',
                    '@Monoclops37', '@dwhite612', '@_bwright', '@InsaneGamer1983', '@avi_ranganath', '@Karthik81422020',
                    '@irina3529', '@Samaritan101011', '@SahanaShreedhar']

    # Define list of verified users. Will use these accounts as confirmed 'non-bot's.
    v_user_list = ['@BarackObama', '@rihanna', '@realDonaldTrump', '@secupp', '@ChairmanKimNK',
                   '@taylorswift13', '@ladygaga', '@TheEllenShow', '@Cristiano', '@YouTube', '@katyperry',
                   '@jtimberlake', '@KimKardashian', '@ArianaGrande', '@britneyspears', '@cnnbrk', '@BillGates',
                   '@narendramodi', '@Oprah', '@SecPompeo', '@nikkihaley', '@SamSifton', '@FrankBruni',
                   '@The_Hank_Poggi', '@krassenstein', '@TheJordanRachel', '@MrsScottBaio',
                   '@ClaireBerlinski', '@java', '@JakeSherman', '@jaketapper', '@jakeowen', '@AndrewCMcCarthy',
                   '@tictoc', '@thedailybeast', '@mitchellvii', '@GadSaad', '@Joy_Villa', '@RashanAGary',
                   '@DallasFed', '@Gab.ai', '@bigleaguepolitics', '@Circa', '@EmilyMiller', '@francesmartel',
                   '@andersoncooper', '@nico_mueller', '@NancyGrace', '@washingtonpost', '@ThePSF', '@pnut',
                   '@EYJr', '@MCRofficial', '@RM_Foundation', '@tomwaits', '@burbunny', '@justinbieber',
                   '@TherealTaraji', '@duttypaul', '@AvanJogia', '@AlecJRoss', '@s_vakarchuk', '@elongmusk',
                   '@StephenColletti', '@jem', '@tonyparker', '@vitorbelfort', '@jeff_green22',
                   '@TomJackson57', '@robbiewilliams', '@AshleyMGreene', '@edhornick', '@mattdusk',
                   '@ReggieEvans30', '@RachelNichols1', '@AndersFoghR', '@PalmerReport',
                   '@KAKA', '@Robbie_OC', '@josiahandthe', '@OKKenna', '@CP3', '@crystaltamar',
                   '@MichelleDBeadle', '@Jonnyboy77', '@kramergirl', '@johnwoodRTR', '@StevePeers',
                   '@AdamSchefter', '@georgelopez', '@CharlieDavies9', '@Nicole_Murphy',
                   '@vkhosla', '@NathanPacheco', '@SomethingToBurn', '@jensstoltenberg', '@Devonte_Riley',
                   '@FreddtAdu', '@Erik_Seidel', '@Pamela_Brunson', '@MMRAW', '@russwest44', '@shawnieora',
                   '@wingoz', '@ToddBrunson', '@NathanFillion', '@LaurenLondon', '@francescadani',
                   '@howardhlederer', '@MrBlackFrancis', '@GordonKljestan', '@thehitwoman', '@KeriHilson',
                   '@druidDUDE', '@jimjonescapo', '@myfamolouslife', '@PAULVANDYK', '@SteveAustria',
                   '@bandofhoreses', '@jaysean', '@justdemi', '@MaryBonoUSA', '@PaulBrounMD', '@jrich23', '@Eve6',
                   '@st_vincent', '@Padmasree', '@jamiecullum', '@GuyKawasaki', '@PythonJones', '@sffed',
                   '@howardlindzon', '@xonecole', '@AlisonSudol', '@SuzyWelch', '@topchefkevin', '@MarcusCooks',
                   '@Rick_Bayless', '@ShaniDavis', '@scottylago', '@danielralston', '@crystalshawanda',
                   '@TheRealSimonCho', '@ItsStephRice', '@IvanBabikov', '@DennyMdotcom', '@TFletchernordic',
                   '@RockneBru86', '@JuliaMancuso', '@RyanOBedford', '@speedchick428', '@JennHeil',
                   '@katadamek', '@kathryn_kang', '@alejandrina_gr', '@RaymondArroyo', '@JonHaidt',
                   '@DKShrewsbury', '@faisalislam', '@miqdaad', '@michikokakutani', '@mehdirhasan', '@AbiWilks',
                   '@hugorifkind', '@kylegriffin1', '@timothy_stanley', '@NAXWELL', '@PT_Dawson', '@MaiaDunphy',
                   '@zachheltzel', '@KatyWellhousen', '@NicholasHoult', '@ryanbroems', '@LlamaGod', '@boozan',
                   '@DarrenMattocks', '@BraulioAmado', '@bernierobichaud', '@ThisisSIBA', '@Jill_Perkins3',
                   '@D_Breitenstein', '@George_McD', '@RedAlurk', '@NickRobertson10', '@kevinvu', '@Henry_Kaye',
                   '@Chris_Biele', '@tom_watson', '@MikeSegalov', '@edballs', '@TalbertSwan', '@eugenegu',
                   '@Weinsteinlaw', '@BrittMcHenry', '@ava', '@McFaul', '@DaShanneStokes', '@funder',
                   '@BrunoAmato_1', '@DirkBlocker', '@TrevDon', '@DavidYankovich', '@KirkDBorne', '@JohnLegere',
                   '@JustinPollard', '@MattDudek', '@CoachWash56', '@RexxLifeRaj', '@SageRosenfels18']

    if os.path.exists("project-samaritan1011001/osna/data/social_spambots_1.csv") and os.path.exists("project-samaritan1011001/osna/data/genuine_accounts.csv"):
        print(f'COLLECTION STARTED\n')
        start = time.time()
        print()
        print(f'COLLECTING BOTS FROM LOCAL DATASET AT osna/data/social_spambots_1.csv\n')
        twitter.fetch_bot_dataset_and_store()
        print(f'FINISHED COLLECTING BOTS AND STORED IN {directory}\n')
        print(f'========================================================================================================\n')
        print(f'COLLECTING GENUINE USERS FROM LOCAL DATASET AT osna/data/genuine_accounts.csv\n')
        twitter.fetch_genuine_dataset_and_store()
        print(f'FINISHED COLLECTING USERS AND STORED IN {directory}\n')
        print(f'========================================================================================================\n')
        print("Number of verified users to be collected from twitter: {}".format(len(v_user_list)))
        twitter.fetch_v_user_and_store(v_user_list,1)
        print(f'FINISHED COLLECTING VERIFIED USERS AND STORED IN {directory}\n')
        print(f'========================================================================================================\n')
        print("Number of non-verified users to be collected from twitter: {}".format(len(nv_user_list)))
        twitter.fetch_nv_user_and_store(nv_user_list,1)
        print(f'FINISHED COLLECTING UNVERIFIED USERS AND STORED IN {directory}\n')
        print(f'========================================================================================================\n')
        print(f'MERGING ALL DATA\n')
        twitter.merge_bot_user_datasets()
        print(f'TOTAL TIME TAKEN TO COLLECT -> {time.time() - start} \n')
        print()
        print(f'COLLECTION COMPLETE and DATA STORED IN {directory}\n')
    else:
        print(f' Make sure files social_spambots_1.csv and genuine_accounts.csv are in project-samaritan1011001/osna/data\n')

@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics of your approach.
    For example, compare classification accuracy for different
    methods.
    """
    # Report accuracies taken from the train method
    clf, accuracies = pickle.load(open(clf_path, 'rb'))
    accuracies_df = pd.DataFrame(accuracies)
    index_list = accuracies_df.index.tolist()
    index_list[0] = 'Mean'
    index_list[1] = 'Standard Deviation'
    accuracies_df.index = index_list
    print(f'\nMEAN ACCURACIES FOR ALL CLASSIFIERS\n')
    print(tabulate(accuracies_df, headers='keys', tablefmt='psql'))


@main.command('network')
def network():
    """
    Perform the network analysis component of your project.
    E.g., compute network statistics, perform clustering
    or link prediction, etc.
    """
    le = LabelEncoder()

    twitter = Twitter(credentials_path)
    features = ['name', 'screen_name', 'description', 'location',
                'friends_count', 'favourites_count', 'followers_count', 'listed_count',
                'statuses_count', 'has_bio', 'followers_count_gr_30', 'bot_in_biography',
                'no_location', 'followers_2_times_ge_friends', 'ratio_friends_followers_around_100']

    influential_users = ['@realDonaldTrump', '@narendramodi']
    print(f'FINDING COMMON NEIGHBORS FOR {influential_users}\n')
    common_neighbors, edges_list_dict = twitter.findCommonNeighbors(influential_users)
    print(f'NUMBER OF COMMON NEIGHBORS FOUND-> {len(common_neighbors)}\n')

    print(f'Fetching Data for common neighbors')
    data = pd.read_json(twitter.fetch_test_data(common_neighbors, 1))
    print(f'NUMBER OF USERS COLLECTED -> {len(data)}')
    collected_screen_names = list(data["screen_name"])

    # Convert null categorical values to missing or if present, transform using labelencoder
    categorical = list(data.select_dtypes(include=['object']).columns.values)
    for cat in categorical:
        data[cat].fillna('missing', inplace=True)
        data[cat] = le.fit_transform(data[cat].astype(str))
    clf, accuracies = pickle.load(open(clf_path, 'rb'))
    result = clf.predict(data[features])
    pretty_print = dict(zip(collected_screen_names,list(result)))
    pretty_print_df = pd.DataFrame(pretty_print, index=["Predicted Values"])
    print(tabulate(pretty_print_df, headers='keys', tablefmt='psql'))
    colors = ['blue'] * (len(common_neighbors) - len(result))
    for res in result:
        if res:
            colors.append('red')
        else:
            colors.append('blue')
    colors.append('blue')
    colors.append('blue')

    # Plot graph for the common neighbors found.
    graph = nx.Graph()
    graph.add_edges_from(edges_list_dict)
    graph.add_edge("@realDonaldTrump", "@narendramodi", len="6.0", color="green", width=10000)
    nx.draw_networkx(graph, with_labels=True, node_color=colors)
    plt.axis('off')
    plt.savefig('project-samaritan1011001/osna/data/graphs/network.png')
    plt.show()
    print(f'GRAPH saved to project-samaritan1011001/osna/data/graphs/network.png\n')



@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all data and print statistics.
    E.g., how many messages/users, time range, number of terms/tokens, etc.
    """
    if os.path.exists(directory + os.path.sep + 'final_user_master.json'):
        print(f'Plotting graphs for the data collected from {directory + os.path.sep + "final_user_master.json"}')
        with open(directory + os.path.sep + 'final_user_master.json') as json_file:
             user_json = json.load(json_file)

        graph_directory = "project-samaritan1011001/osna/data/graphs/"
        data = pd.read_json(user_json)

        # Prints various graphs for the data collected.
        get_heatmap(data, graph_directory)
        plot_freinds_followers_graph(directory, graph_directory)
        another_heat_map(data, graph_directory)
        plot_all_bots_freinds_followers(directory,graph_directory)
        plot_real_users_freinds_followers(directory,graph_directory)

        print(f'Graphs saved to project-samaritan1011001/osna/data/graphs/\n')
    else:
        print(f'Run Collect script with the same directory first to show stats\n')


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier on all of your labeled data and save it for later
    use in the web app. You should use the pickle library to read/write
    Python objects to files. You should also reference the `clf_path`
    variable, defined in __init__.py, to locate the file.
    """

    if os.path.exists(directory + os.path.sep + 'final_user_master.json'):
        start = time.time()

        le = LabelEncoder()

        # Classifier Intialization
        lr_clf = LogisticRegression(solver='lbfgs',max_iter=3000)

        dt_clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

        rf_clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20,
                                                         n_estimators=100)

        with open(directory + os.path.sep + 'final_user_master.json') as json_file:
            user_json = json.load(json_file)
        data = pd.read_json(user_json)
        data = data.reindex(np.random.permutation(data.index))
        data = pd.DataFrame(data).fillna(0)
        categorical = list(data.select_dtypes(include=['object']).columns.values)
        for cat in categorical:
            data[cat].fillna('missing', inplace=True)
            data[cat] = le.fit_transform(data[cat].astype(str))
        lr_accuracies = []
        dt_accuracies = []
        rf_accuracies = []

        print(f'TRAINING DATA FROM {directory + os.path.sep + "final_user_master.json"} with {len(data)} users\n')

        # KFold implementation with number of splits = 5, used for cross-validation
        k = KFold(n_splits=5)
        for result in k.split(data):
            X_train = data.iloc[result[0]]
            X_test = data.iloc[result[1]]

            # Delete column known_bot so we don't train with the truth values
            del X_train['known_bot']
            del X_test['known_bot']
            # Truth values for both train and test data
            y_train = data.iloc[result[0]]['known_bot']
            y_test = data.iloc[result[1]]['known_bot']

            # Train the classifiers
            lr_clf.fit(X_train, y_train)
            dt_clf.fit(X_train, y_train)
            rf_clf.fit(X_train, y_train)

            # Record the accuracies given by the score function
            lr_accuracies.append(lr_clf.score(X_test, y_test))
            dt_accuracies.append(dt_clf.score(X_test, y_test))
            rf_accuracies.append(rf_clf.score(X_test, y_test))

        print(f'TRAINING FINISHED\n')
        print(f'TOTAL TIME TAKEN TO TRAIN -> {time.time() - start}')
        print(f'mean lr_accuracy -> {np.mean(lr_accuracies)} and std -> {np.std(lr_accuracies)}')
        print(f'mean dt_accuracy -> {np.mean(dt_accuracies)} and std -> {np.std(dt_accuracies)}')
        print(f'mean rf_accuracy -> {np.mean(rf_accuracies)} and std -> {np.std(rf_accuracies)}')
        accuracies = {"Logistic Regression":[np.mean(lr_accuracies),np.std(lr_accuracies)],"Decision Trees":[np.mean(dt_accuracies),np.std(dt_accuracies)],
                      "Random Forest":[np.mean(rf_accuracies),np.std(rf_accuracies)]}

        # Dump the selected Decision tree classifier using pickle at clf_path
        pickle.dump((dt_clf,accuracies), open(clf_path, 'wb'))
        print(f'DUMPED DECISION CLASSIFIER USING PICKLE\n')
    else:
        print(f'Make sure collect script is run using the same directory\n')



@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='127.0.0.1', debug=True, port=port)


# Helper function to plot all the graphs
def plot_final_users_freinds_followers(data, directory):
    plt.xlabel('screen_name', size=14)
    plt.ylabel('followers_count', size=14)
    plt.plot(data['followers_count'])
    plt.savefig(directory  + "final_users_freinds_followers.png")
    plt.show()

def plot_all_bots_freinds_followers(directory,graph_directory):
    with open(directory + os.path.sep + 'b_user_table_out.json') as json_file:
        user_json = json.load(json_file)
    data = pd.read_json(user_json)
    plt.xlabel('bots count', size=14)
    plt.ylabel('followers_count', size=14)
    plt.plot(data['followers_count'])
    plt.savefig(graph_directory + "all_bots_freinds_followers.png")
    plt.show()


def plot_real_users_freinds_followers(directory,graph_directory):
    with open(directory + os.path.sep + 'v_user_table_out.json') as json_file:
        v_user_json = json.load(json_file)
    with open(directory + os.path.sep + 'nv_user_table_out.json') as json_file:
        nv_user_json = json.load(json_file)
    with open(directory + os.path.sep +  'g_user_table_out.json') as json_file:
        g_user_json = json.load(json_file)
    v_user_df = pd.read_json(v_user_json)
    nv_user_df = pd.read_json(nv_user_json)
    g_user_df = pd.read_json(g_user_json)
    plt.xlabel('User count', size=14)
    plt.ylabel('followers_count', size=14)
    plt.plot(v_user_df['followers_count'],color='blue',label='Verified user Followers')
    plt.plot(nv_user_df['followers_count'],color='red',label='Non-verified Followers')
    plt.plot(g_user_df['followers_count'],color='green',label='Real users Followers')
    plt.legend(loc='upper left')
    plt.savefig(graph_directory + "real_users_freinds_followers.png")
    plt.show()


def another_heat_map(df, graph_directory):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(method='spearman'), cmap='coolwarm', annot=True)
    plt.savefig(graph_directory + "another_heatmap.png")
    plt.show()


def get_heatmap(df,graph_directory):
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.tight_layout()
    plt.savefig(graph_directory  +"heatmap.png")
    plt.show()


def plot_freinds_followers_graph(directory,graph_directory):
    with open(directory + os.path.sep + 'b_user_table_out.json') as json_file:
        user_json = json.load(json_file)
    bots = pd.read_json(user_json)
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.title('Bots Friends vs Followers')
    sns.regplot(bots["friends_count"], bots["followers_count"], color='red', label='Bots')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    with open(directory + os.path.sep + 'v_user_table_out.json') as json_file:
        v_user_json = json.load(json_file)
    with open(directory + os.path.sep + 'nv_user_table_out.json') as json_file:
        nv_user_json = json.load(json_file)
    with open(directory + os.path.sep + 'g_user_table_out.json') as json_file:
        g_user_json = json.load(json_file)
    v_user_df = pd.read_json(v_user_json)
    nv_user_df = pd.read_json(nv_user_json)
    g_user_df = pd.read_json(g_user_json)
    # Merging v_users and nv_users
    user_df = nv_user_df.append(v_user_df,sort=False)
    user_df = user_df.append(g_user_df,sort=False)
    plt.subplot(2, 1, 2)
    plt.title('Real users\' Friends vs Followers')
    sns.regplot(user_df["friends_count"], user_df["followers_count"], color='blue', label='NonBots')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(graph_directory  + "freinds_followers.png")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
