# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # 设置你的 CSV 文件所在目录
# csv_folder = '/home/v100/tbl/Time-Series-Library/dataset/google'  # ← 修改为你的目录路径

# # 获取该目录下所有 .csv 文件
# csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# # 遍历每个 CSV 文件
# for file in csv_files:
#     file_path = os.path.join(csv_folder, file)

#     # 读取数据
#     df = pd.read_csv(file_path, parse_dates=['date'])
#     df['date'] = pd.to_datetime(df['date'], unit='us')

#     # 画图
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['date'], df['avgcpu'], label='avgcpu', color='blue')
#     plt.plot(df['date'], df['avgmem'], label='avgmem', color='orange')
    
#     plt.title(f'Trend of avgcpu and avgmem - {file}')
#     plt.xlabel('Time')
#     plt.ylabel('Usage')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # 可选择保存图像，也可以直接展示
#     plt.savefig(f"{file}_trend.png")  # 如果你想保存图像
#     # plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
'''
ali18.csv:2018-05-09~2018-05-16
ali20_*.csv:2020-07-15~2020-09-01
gc11.csv:2011-05-01~2011-05-29
gc19_*.csv:2019-05-01~2019-05-29
'''


csv_folder = '/home/v100/tbl/Time-Series-Library/dataset/google'
# csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
# csv_files = '/home/v100/tbl/Time-Series-Library/dataset/alibaba/ali18.csv'
# csv_files = ['ali18.csv']
# csv_files = ['ali20_c.csv','ali20_g.csv']
# csv_files = ['gc11.csv']
csv_files = ['gc19_a.csv','gc19_b.csv','gc19_c.csv','gc19_d.csv','gc19_e.csv','gc19_f.csv','gc19_g.csv','gc19_h.csv']

# 设定统一的时间段（你可以改）
# start_time = pd.to_datetime('2018-05-13 00:00:00')
# end_time = pd.to_datetime('2018-05-15 00:00:00')
# start_time = pd.to_datetime('2020-08-30 00:00:00')
# end_time = pd.to_datetime('2020-09-01 00:00:00')
# start_time = pd.to_datetime('2011-05-26 00:00:00')
# end_time = pd.to_datetime('2011-05-28 00:00:00')
start_time = pd.to_datetime('2019-05-20 00:00:00')
end_time = pd.to_datetime('2019-05-22 00:00:00')
for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], unit='us')

    # 筛选时间段
    df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]

    # 如果该段数据太少就跳过
    if len(df) < 10:
        continue

    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['avgcpu'], label='avgcpu', color='blue')
    plt.plot(df['date'], df['avgmem'], label='avgmem', color='orange')
    
    plt.title(f'{file} - avgcpu & avgmem from {start_time.date()} to {end_time.date()}')
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # 可选择保存图像，也可以直接展示
    plt.savefig(f"{file}_{start_time}_{end_time}_trend.png")  # 如果你想保存图像
