import requests
import json
import pandas


def get_coord(domen):
    """получение координат и id точек выдачи"""
    url = f'https://{domen}/webapi/spa/modules/pickups'
    headers = {'User-Agent': "Mozilla/5.0", 'content-type': "application/json", 'x-requested-with': 'XMLHttpRequest'}
    r = requests.get(url, headers=headers)
    data = r.json()
    
    data_list = []
    for d in data['value']['pickups']:
        id = d['id']
        address = d['address']
        coordinates = d['coordinates']
        workTime = d.get('workTime', '')
        data_list.append({
            'id': int(id),
            'address': address,
            'coordinates': coordinates,
            'workTime': workTime
        })
    print("[INFO] координаты точек выдачи получены")
    return data_list


def get_points(payload: list, domen: str):
    """получаем данные по всем пунктам выдачи"""
    url = f"https://{domen}/webapi/poo/byids"
    headers = {'User-Agent': "Mozilla/5.0", 'content-type': "application/json"}
    
    data_points_list = []

    ## Разделим payload на части по 500 для избежания перегрузки
    batch_size = 500
    for i in range(0, len(payload), batch_size):
        batch_payload = payload[i:i + batch_size]
        payload_str = f'{batch_payload}'  ## Формируем строку с payload для POST запроса
        
       
        response = requests.post(url, data=payload_str, headers=headers)
        data = response.json()

  
        for d in data['value']:
            wayinfo = data['value'][d].get('wayInfo', '')
            rate = data['value'][d].get('rate', 0)
            data_points_list.append({
                'id': int(d),
                'rate': rate,
                'wayInfo': wayinfo.replace('\n', ' ')
            })

    print("[INFO] данные по всем точкам выдачи получены")
    return data_points_list


def merge_data(data1: list, data2: list):
    """объединение таблиц с помощью датафреймов"""
    df1 = pandas.DataFrame(data1)
    df2 = pandas.DataFrame(data2)

    df = pandas.merge(df1, df2, how='left', left_on='id', right_on="id")
    df = df[['id', 'rate', 'address', 'workTime', 'wayInfo', 'coordinates']]

    ## Запись в json файл
    df.to_json('NEW_wb_points.json', orient='records', force_ascii=False, index=False)

    ## Запись в Excel файл
    writer = pandas.ExcelWriter('NEW_wb_points.xlsx', engine='xlsxwriter')
    df.to_excel(writer, 'data')
    writer._save()

    print(f'[INFO] Данные объединены и сохранены в NEW_wb_points.xlsx\n'
          f'[INFO] Количество найденных пунктов выдачи: {len(df)}\n'
          f'Работа парсера завершена')


def main(domen):
    """получаем все координаты и id пунктов выдачи"""
    data_list_coords = get_coord(domen=domen)

    """собираем все ID для запроса данных"""
    payload_generator = [i['id'] for i in data_list_coords]

    """получаем данные по всем точкам выдачи"""
    data_list_points = get_points(payload=payload_generator, domen=domen)

    """объединяем данные и сохраняем их в файлы"""
    merge_data(data1=data_list_points, data2=data_list_coords)


if __name__ == '__main__':
    main('www.wildberries.ru')