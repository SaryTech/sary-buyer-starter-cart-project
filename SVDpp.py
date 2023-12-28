import pandas as pd
import numpy as np
import pickle
import snowflake.connector as sf
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import settings
import requests
import ast
import json

class SnowflakeConnection:
    def __init__(self):
        self.config = {
            'SNOWFLAKE_ACCOUNT': settings.SNOWFLAKE_ACCOUNT,
            'SNOWFLAKE_USER': settings.SNOWFLAKE_USER,
            'SNOWFLAKE_PASSWORD': settings.SNOWFLAKE_PASSWORD,
            'SNOWFLAKE_WAREHOUSE': settings.SNOWFLAKE_WAREHOUSE,
            'SNOWFLAKE_ROLE': settings.SNOWFLAKE_ROLE,
            'DATABASE': settings.SNOWFLAKE_DATABASE,
            'SCHEMA': settings.SNOWFLAKE_SCHEMA
        }

    def establish_connection(self):
        return sf.connect(
            account=self.config['SNOWFLAKE_ACCOUNT'],
            user=self.config['SNOWFLAKE_USER'],
            password=self.config['SNOWFLAKE_PASSWORD'],
            database=self.config['DATABASE'],
            schema=self.config['SCHEMA'],
            role=self.config['SNOWFLAKE_ROLE'],
            warehouse=self.config['SNOWFLAKE_WAREHOUSE']
        )

    def fetch_data(self, query):
        conn = self.establish_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            return pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
        finally:
            cursor.close()
            conn.close()

    def fetch_excluded_users_data(self):
        query = f"""select user_id, business_segment, orders_with_financials.city_name, orders_with_financials.order_id, orders_orderproduct.package_id as item_id
from users_info
join orders_with_financials using(user_id)
join orders_orderproduct on orders_orderproduct.order_id = orders_with_financials.order_id
where users_info.user_id in(
55884, 201967, 216897, 179825, 386398, 147881, 158900, 240915, 186836, 164215, 29061,
                             101826, 241170, 388290, 74169, 18161, 329995, 414840, 78096, 248745, 142870, 66750, 149136,
                             190716, 80230, 118538, 53066, 275240, 149631, 421189
)"""
        return self.fetch_data(query)

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def time_decay(self, timestamp, decay_rate):
        return np.exp(-decay_rate * timestamp)

    def preprocess(self):
        self.data = self.data[~self.data['BUSINESS_SEGMENT'].isin(['Testing Environment - For the content team', 'Sary Van'])]
        self.data.dropna(subset=['BUSINESS_SEGMENT', 'CITY_NAME'], inplace=True)
        self.data[['ITEM_ID', 'USER_ID']] = self.data[['ITEM_ID', 'USER_ID']].astype(int)
        self.data['BUSINESS_SEGMENT'] = self.data['BUSINESS_SEGMENT'].apply(
            lambda x: 'Groceries' if x == 'Groceries - Snacks' else (
                'Wholesalers' if x in ['Semi Wholesaler', 'Wholesalers (Flagship)', 'Wholesalers & Retailers', 'Wholesalers (Flagship)'] else x))
        self.data = self.data[~self.data['CITY_NAME'].isin(['Cairo', 'Qalyubiya', 'Giza'])]

        self.data[['CITY_NAME', 'DISTRICT_NAME', 'EVENT_TYPE', 'BUSINESS_SEGMENT']] = self.data[
            ['CITY_NAME', 'DISTRICT_NAME', 'EVENT_TYPE', 'BUSINESS_SEGMENT']].apply(lambda x: x.str.lower().str.strip())

        event_type_to_rating = {
            'Purchase': 5,
            'View': 3,
            'add-to-cart': 4,
            'Search': 2
        }
        self.data['RATING'] = self.data['EVENT_TYPE'].map(event_type_to_rating)
        decay_rate = 1e-9
        self.data['EVENT_TYPE_DECAYED'] = self.data['RATING'] * self.data.apply(lambda x: self.time_decay(x['TIMESTAMP'], decay_rate), axis=1)

        #randomly excluded users
        excluded_user_ids = [55884, 201967, 216897, 179825, 386398, 147881, 158900, 240915, 186836, 164215, 29061, 101826, 241170, 388290, 74169, 18161, 329995, 414840, 78096, 248745, 142870, 66750, 149136, 190716, 80230, 118538, 53066, 275240, 149631, 421189]
        self.data = self.data[~self.data['USER_ID'].isin(excluded_user_ids)]
        return self.data

class RecommenderSystem:
    def __init__(self, data):
        self.data = data

    def train_model(self):
        min_rating = min(self.data['EVENT_TYPE_DECAYED'])
        max_rating = max(self.data['EVENT_TYPE_DECAYED'])
        reader = Reader(rating_scale=(min_rating, max_rating))
        dataset = Dataset.load_from_df(self.data[['USER_ID', 'ITEM_ID', 'RATING']], reader)
        trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

        model = SVDpp()
        model.fit(trainset)
        return model, testset

    def save_model(self, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def evaluate_model(self, model, testset):
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f"RMSE: {rmse}")

    def recommend_items_to_new_user(self, business_segment, city_name, loaded_model):
        filtered_data = self.data[(self.data['BUSINESS_SEGMENT'] == business_segment) & (self.data['CITY_NAME'] == city_name)]
        purchased_items = filtered_data['ITEM_ID'].unique()

        recommendations = []
        for item_id in purchased_items:
            prediction = loaded_model.predict(uid='new_user', iid=item_id)
            recommendations.append((item_id, prediction.est))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in recommendations[:10]]

    def model_exists(self, path):
        try:
            with open(path, 'rb') as file:
                pickle.load(file)
            return True
        except FileNotFoundError:
            return False

class Main:
    def __init__(self):
        self.snowflake_conn = SnowflakeConnection()

    def run(self):
        recommender = RecommenderSystem(None)
        sql_query = """select distinct base2.*, users_info.business_segment, orders_with_financials.city_name, orders_with_financials.district_name
                    from 
                    (
                       select
                        base.*
                        from
                          (
                            -- For SKUs added to the cart using normal method
                            select
                              COALESCE(user_id,0) as USER_ID,
                              COALESCE(package_id,0) as ITEM_ID,
                              date_part(epoch_second, timestamp) as timestamp,
                              --converted to UNIX epoch time format
                              -- original_timestamp,
                              'add-to-cart' AS EVENT_TYPE
                              -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                            from
                              segment_events.ios_development.sku_add_to_cart
                            where
                              to_char(timestamp, 'YYYY') >= '2023'
                              and user_id is not null and user_id != 'null'
                            UNION(
                                -- For SKUs added to the cart using quick add method
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'add-to-cart' AS EVENT_TYPE
                                  -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                                from
                                  segment_events.ios_development.category_item_quick_add_to_cart
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  -- For SKUs viewed
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'View' AS EVENT_TYPE
                                from
                                  segment_events.android.sku_viewed
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  -- For SKUs viewed
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'View' AS EVENT_TYPE
                                from
                                  segment_events.ios_development.sku_viewed
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                -- For SKUs added to the cart through search using quick add method
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'search' AS EVENT_TYPE
                                  -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                                from
                                  segment_events.ios_development.search_item_quick_add_to_cart
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                -- For SKUs added to the cart using normal method
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'add-to-cart' AS EVENT_TYPE
                                  -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                                from
                                  segment_events.android.sku_add_to_cart
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                -- For SKUs added to the cart using quick add method
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'add-to-cart' AS EVENT_TYPE
                                  -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                                from
                                  segment_events.android.category_item_quick_add_to_cart
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                -- For SKUs added to the cart through search using quick add method
                                select
                                  COALESCE(user_id,0) as USER_ID,
                                  COALESCE(package_id,0) as ITEM_ID,
                                  date_part(epoch_second, timestamp) as timestamp,
                                  --converted to UNIX epoch time format
                                  -- original_timestamp,
                                  'search' AS EVENT_TYPE
                                  -- CONTEXT_TRAITS_BUSINESS_SEGMENT as business_segment
                                from
                                  segment_events.android.search_item_quick_add_to_cart
                                where
                                  to_char(timestamp, 'YYYY') >= '2023'
                                  and user_id is not null and user_id != 'null'
                              )
                            UNION(
                                select
                                  COALESCE(orders_with_financials.user_id, 0) as USER_ID, -- Replace NULL user_id with -1
                                COALESCE(COALESCE(package_id, 0), 0) as ITEM_ID,
                                  date_part(epoch_second, order_date) as TIMESTAMP,
                                  'Purchase' as EVENT_TYPE -- , business_segment
                                from
                                  sary_hevo.public.orders_with_financials
                                  join sary_hevo.public.orders_shipment using(order_id)
                                  join sary_hevo.public.orders_orderproduct on orders_shipment.id = orders_orderproduct.shipment_id
                                  join sary_hevo.public.verticals_branch on orders_shipment.branch_id = verticals_branch.id
                                  join sary_hevo.public.users_info on orders_with_financials.user_id = users_info.user_id
                                where
                                  is_for_business = false
                                  and to_char(order_date, 'YYYY') >= '2023'
                                  and orders_with_financials.user_id is not null
                              )
                          ) as base
                          order by 1 desc,3
                    )as base2 
                    join users_info on base2.user_id = users_info.user_id
                    join sary_hevo.public.orders_with_financials on users_info.user_id = orders_with_financials.user_id
                    where to_char(users_info.last_order_time, 'YYYY') = '2023'
                    AND EVENT_TYPE = 'Purchase'"""
        data = self.snowflake_conn.fetch_data(sql_query)

        processor = DataProcessor(data)
        processed_data = processor.preprocess()
        # User prompt to decide whether to train the model
        train_model = input("Do you want to train the model from the beginning? (yes/no): ").lower().strip()
        recommender = RecommenderSystem(processed_data)

        if train_model == 'yes' or not recommender.model_exists('trained_model.pkl'):
            recommender = RecommenderSystem(processed_data)
            model, testset = recommender.train_model()
            recommender.save_model(model, 'trained_model.pkl')
        else:
            # Load existing model
            loaded_model = recommender.load_model('trained_model.pkl')
            data = self.snowflake_conn.fetch_data(sql_query)
            processor = DataProcessor(data)
            processed_data = processor.preprocess()
            recommender = RecommenderSystem(processed_data)

        # Fetching and processing data for excluded users
        excluded_users_data = self.snowflake_conn.fetch_excluded_users_data()
        excluded_users_data['RATING'] = 5
        excluded_users_data.rename(columns={'business_segment': 'BUSINESS_SEGMENT', 'city_name': 'CITY_NAME'},
                                   inplace=True)
        excluded_users_data[['BUSINESS_SEGMENT', 'CITY_NAME']] = excluded_users_data[
            ['BUSINESS_SEGMENT', 'CITY_NAME']].apply(lambda x: x.str.lower().str.strip())

        excluded_users_data = excluded_users_data[['USER_ID', 'BUSINESS_SEGMENT', 'CITY_NAME', 'ORDER_ID','ITEM_ID','RATING']]
        excluded_users_data.drop_duplicates(inplace=True)
        excluded_users_data = excluded_users_data.groupby(['USER_ID', 'ORDER_ID', 'BUSINESS_SEGMENT', 'CITY_NAME', 'RATING'])[
            'ITEM_ID'].unique().reset_index()
        excluded_users_data.rename(columns={'ITEM_ID': 'unique_item_ids'}, inplace=True)
        unique_pairs = excluded_users_data[['CITY_NAME', 'BUSINESS_SEGMENT']].drop_duplicates()

        # Generate recommendations for excluded users
        predictions_df = pd.DataFrame(columns=['CITY_NAME', 'BUSINESS_SEGMENT', 'RECOMMENDED_ITEMS'])
        for index, row in unique_pairs.iterrows():
            city_name = row['CITY_NAME']
            business_segment = row['BUSINESS_SEGMENT']
            recommended_items = recommender.recommend_items_to_new_user(business_segment, city_name, loaded_model)
            new_row = pd.DataFrame({'CITY_NAME': [city_name], 'BUSINESS_SEGMENT': [business_segment],
                                    'RECOMMENDED_ITEMS': [recommended_items]})
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

        merged_df = pd.merge(excluded_users_data, predictions_df, on=['CITY_NAME', 'BUSINESS_SEGMENT'])
        merged_df.drop_duplicates(subset=['USER_ID', 'BUSINESS_SEGMENT', 'CITY_NAME'], inplace=True)
        final_df = merged_df[['USER_ID', 'BUSINESS_SEGMENT', 'CITY_NAME', 'RECOMMENDED_ITEMS', 'unique_item_ids']]
        print(final_df)


        def calculate_hit_rate(recommended_items, unique_items):
            if not list(unique_items):
                return 0
            hit_count = sum(item in recommended_items for item in unique_items)
            return hit_count / len(list(unique_items)) if list(unique_items) else 0

        # final_df['unique_item_ids'] = final_df['unique_item_ids'].apply(adjust_convert_to_list_v2)
        # HitRate Calculation
        final_df['HIT_RATE'] = final_df.apply(lambda row: calculate_hit_rate(row['RECOMMENDED_ITEMS'], row['unique_item_ids']), axis=1)

        def get_similar_items(item_id, user_id):
            url = 'https://europe-west1-data-team-general-purpose.cloudfunctions.net/SimilarItemsRecommendationEngine'
            headers = {
                'Authorization': 'bearer {}',  # Add your bearer token here
                'Content-Type': 'application/json'
            }
            data = {"userId": str(user_id), "itemId": str(item_id), "quantity": "10", "numResults": "20"}
            response = requests.post(url, headers=headers, json=data, timeout=100)
            if response.status_code == 200:
                recommended_items = response.json().get('recommendedItems', [])
                # Parse the string to a list and then convert each item to an integer
                return [int(item) for item in ast.literal_eval(recommended_items)]
            else:
                return []

        def get_all_similar_items(unique_item_ids, user_id):
            similar_items = []
            for item_id in unique_item_ids:
                similar_items.extend(get_similar_items(item_id, user_id))
            return list(set(similar_items))

        final_df['SIMILAR_ITEMS'] = final_df.apply(lambda row: get_all_similar_items(row['unique_item_ids'], row['USER_ID']),axis=1)
        final_df['HIT_RATE_SIMILAR_ITEMS'] = final_df.apply(
            lambda row: calculate_hit_rate(row['RECOMMENDED_ITEMS'], row['SIMILAR_ITEMS']), axis=1)
        final_df.to_csv('test.csv', index=False)
        print(final_df.head())

if __name__ == "__main__":
    main_app = Main()
    main_app.run()

#WIP - The model invocation has some bugs that need to be fixed prior to calculating the Hit Rate
# The invocation under invoking_model.py works fine