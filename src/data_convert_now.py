'''
Created on Jan 23, 2025

@author: Admin
'''

import os
import json
import requests
import pandas as pd
from datetime import datetime, date, timedelta
import json

end_time = datetime.now()
start_time = end_time - timedelta(days=1)
end_time = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
start_time = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

url = f'https://geomag.usgs.gov/ws/algorithms/filter/?elements=H&format=json&id=BRW&type=adjusted&starttime={start_time}&endtime={end_time}&input_sampling_period=60&output_sampling_period=60'
print(url)
r = requests.get(url)
data_dict = r.json()

folder = r'C:\Users\Admin\eclipse-workspace\magnavis\src'

filename = 'download_mag.json'
with open(os.path.join(folder, filename), 'w') as fp:
    json.dump(data_dict, fp)


'''
returns a dataframe having timeseries of earth's magnetic field
source: https://geomag.usgs.gov/plots/
downloaded_fileformat: json
'''
def get_timeseries_magnetic_data(last_n_samples=None):
    remove_na=True
    files = [filename]
    
    df_final = pd.DataFrame()
    df_list = []
    
    for fil in files:
        full_fname = os.path.join(folder, fil)
        with open(full_fname, 'r') as ff:
            data = json.load(ff)
            orientation = data['metadata']['intermagnet']['reported_orientation']
            print(data['metadata']['intermagnet']['reported_orientation']) #data['reported_orientation'], data['times'])
            times_str = data['times']
            # print(len(times_str))
            times = [datetime.fromisoformat(t[:-1] + '+00:00') for t in times_str]
            # print(times[0])
            mag_field = data['values'][0]['values']
            # print(mag_field[0])
            df_ = pd.DataFrame({
                    f'time_{orientation}': times,
                    f'mag_{orientation}_nT': mag_field
                })
            df_final = pd.concat([df_final, df_], axis=1)
            if remove_na:
                df_final = df_final.dropna()
            if last_n_samples:
                df_final = df_final.dropna().tail(last_n_samples)
    return df_final



if __name__ == '__main__':
    df = get_timeseries_magnetic_data(last_n_samples=10)
    print(df) # datetime, float (field in nT) (NaN supported)
    now = datetime.now().strftime('%Y%m%dT%H%M%S')
    df.astype(str).to_excel(f'magnetic_time_ser_{now}.xlsx', index=False)