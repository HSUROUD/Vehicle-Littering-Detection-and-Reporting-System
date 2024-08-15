import pandas as pd
import MySQLdb


conn = MySQLdb.connect(host="localhost", user="db_project", passwd="databaseproject", db="littering_detection")
cursor = conn.cursor()


df = pd.read_csv('static/8/8.csv')
df['shooting_time'] = pd.to_datetime(df['shooting_time'])
df = df.where(pd.notnull(df), None)

for index, row in df.iterrows():

    cursor.execute('INSERT INTO litter_data (license_plate, garbage_category, frame_number, capture_date, location) VALUES (%s, %s, %s, %s, %s)', 
                   (row['car plate'], row['trash type'], row['frame'], row['shooting_time'], row['shooting_location']))


conn.commit()
conn.close()
