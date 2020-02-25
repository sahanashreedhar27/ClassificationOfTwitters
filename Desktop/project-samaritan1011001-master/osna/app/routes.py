from flask import render_template
from osna.mytwitter import Twitter
from . import app
from .forms import MyForm
from .. import credentials_path, clf_path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import networkx as nx
from matplotlib import pyplot as plt


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    twapi = Twitter(credentials_path)
    clf, accuracies = pickle.load(open(clf_path, 'rb'))
    print('read clf %s' % str(clf))

    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field1 = form.input_field1.data
        input_field2 = form.input_field2.data
        print(input_field1)
        print(input_field2)
        form.common_bots = evaluate(clf, twapi, screen_names=["@" + input_field1, "@" + input_field2])

        return render_template('myform.html', title='Bot Detection', form=form)
    return render_template('myform.html', title='Bot Detection', form=form)


def evaluate(clf, twapi, screen_names):
    le = LabelEncoder()

    features = ['name', 'screen_name', 'description', 'location',
                'friends_count', 'favourites_count', 'followers_count', 'listed_count',
                'statuses_count', 'has_bio', 'followers_count_gr_30', 'bot_in_biography',
                'no_location', 'followers_2_times_ge_friends', 'ratio_friends_followers_around_100']

    common_neighbors, edges_list_dict = twapi.findCommonNeighbors(screen_names)

    graph = nx.Graph()
    graph.add_edges_from(edges_list_dict)
    data = pd.read_json(twapi.fetch_test_data(common_neighbors, 1))
    categorical = list(data.select_dtypes(include=['object']).columns.values)
    for cat in categorical:
        data[cat].fillna('missing', inplace=True)
        data[cat] = le.fit_transform(data[cat].astype(str))
    result = clf.predict(data[features])
    colors = ['blue'] * (len(common_neighbors) - len(result))
    for res in result:
        if res:
            colors.append('red')
        else:
            colors.append('blue')
    colors.append('blue')
    colors.append('blue')
    nx.draw_networkx(graph, with_labels=True, node_color=colors)
    plt.axis('off')
    plt.savefig('project-samaritan1011001/osna/app/static/network.png')
    return data['screen_name']
