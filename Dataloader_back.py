#code claim: 读取数据：使用 pd.read_csv 读取 .log 文件，并指定列名。
#数据清洗：
#使用 dropna 删除包含缺失值的行。
#使用 astype(float) 确保所有数值字段为浮点数类型。
#数据转换：
#使用 math.radians 将 lat 和 lon 转换为弧度。
#使用 pd.to_datetime 将 simt 转换为时间戳，并添加为新的 timestamp 列。
#数据筛选：
#使用布尔索引筛选出 type 为 M600 的记录。
#使用布尔索引筛选出 alt 大于 0 的记录。
#保存处理后的数据：使用 to_csv 将处理后的数据保存为 .csv 文件。
#错误处理：捕获并处理文件未找到、文件为空或格式不正确以及其他未知错误。
#使用方法：
#将 input_file 和 output_file 替换为实际的文件路径。
#运行脚本，处理后的数据将保存到指定的输出文件中

import pandas as pd
import numpy as np
import math

def preprocess_log_file(input_file, output_file):
    try:
        # 读取数据
        df = pd.read_csv(input_file, header=None, names=[
            'simt', 'id', 'type', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs', 
            'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 'drag', 
            'phase', 'fuelflow'
        ])

        # 数据清洗
        # 删除任何包含缺失值的行
        df.dropna(inplace=True)

        # 确保所有数值字段均为浮点数类型
        numeric_columns = ['simt', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs', 
                           'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 
                           'drag', 'fuelflow']
        df[numeric_columns] = df[numeric_columns].astype(float)

        # 数据转换
        # 将 lat 和 lon 转换为弧度
        df['lat'] = df['lat'].apply(math.radians)
        df['lon'] = df['lon'].apply(math.radians)

        # 添加一个新的列 timestamp，表示每个记录的时间戳（假设 simt 是时间戳，单位为秒）
        df['timestamp'] = pd.to_datetime(df['simt'], unit='s')

        # 数据筛选
        # 筛选出 type 为 M600 的记录
        df = df[df['type'] == 'M600']

        # 筛选出 alt 大于 0 的记录
        df = df[df['alt'] > 0]

        # 保存处理后的数据
        df.to_csv(output_file, index=False)

        print(f"数据预处理完成，结果已保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 {input_file} 为空或格式不正确。")
    except Exception as e:
        print(f"发生未知错误：{e}")




# 示例用法
if __name__ == "__main__":
    # 请将 input_file 和 output_file 替换为实际的文件路径
    input_file = 'data.log'  # 输入文件路径
    output_file = 'processed_data.csv'  # 输出文件路径
    preprocess_log_file(input_file, output_file)
