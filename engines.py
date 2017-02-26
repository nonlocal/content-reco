import pandas as pd
import numpy
import re
from flask.json import jsonify
import json
import time
import redis
from flask import current_app, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def info(msg):
    current_app.logger.info(msg)


class ContentEngine(object):

    SIMKEY = 'p:smlr:%s'

    def __init__(self):
        self._r = redis.StrictRedis.from_url(current_app.config['REDIS_URL'])

    def train(self, data_source):
        start = time.time()
        #xxx = pd.ExcelFile("dummy")
        xxx = pd.ExcelFile(data_source)
        ds = xxx.parse("Sheet1")
        #ds = pd.read_csv(data_source)#, error_bad_lines=False)
        info("Training data ingested in %s seconds." % (time.time() - start))

        # Flush the stale training data from redis
        self._r.flushdb()

        start = time.time()
        self._train(ds)
        info("Engine trained in %s seconds." % (time.time() - start))

    def _train(self, ds):
        
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        #myEdits
        xs = ds['Description'].astype(str)
        pattern = re.compile('\W')
        train_sample = [re.sub(pattern, ' ', xyz) for xyz in xs]
        tfidf_matrix = tf.fit_transform(train_sample)#ds['description'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        for idx, row in ds.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], ds['ID'][i]) for i in similar_indices]

            # First item is the item itself, so remove it.
            # This 'sum' is turns a list of tuples into a single tuple: [(1,2), (3,4)] -> (1,2,3,4)
            flattened = sum(similar_items[1:], ())
            self._r.zadd(self.SIMKEY % row['ID'], *flattened)

    def predict(self, item_id, num, data_source):
                
        result = self._r.zrange(self.SIMKEY % item_id, 0, num-1, withscores=True, desc=True)
        ky = []
        for i in range(10):
            ky.append(result[i][0])
        ky = list(map(int, ky))
        xxx = pd.ExcelFile(data_source)
        ds = xxx.parse("Sheet1")
        location = [ds[ds.ID==x].index.tolist() for x in ky]
        location = sum(location, [])
        item_price = list(ds.Price[location])
        item_loc = int(item_id)
        item_loc = (ds.ID==item_loc).argmax()
        sub_price = ds.Price[item_loc]
        adj_price = [abs(x - sub_price) for x in item_price]
        adj_price = numpy.array(adj_price)
        new_index = adj_price.argsort()
        out_price = numpy.array(item_price)
        out_id = numpy.array(ky)
        out_price = out_price[new_index]
        out_id = out_id[new_index]
        #new_keys = [y for (y, x) in sorted(zip(ky, item_price))]
        new_price = map(int, out_price)
        second_arr = list(zip(out_id, new_price))
        abc = [1,2,3,4,5,6,7,8,9,10]
        dikt = list(zip(abc, second_arr))
        return jsonify(dikt) #jsonify(self._r.zrange(self.SIMKEY % item_id, 0, num-1, withscores=True, desc=True))
        
content_engine = ContentEngine()
