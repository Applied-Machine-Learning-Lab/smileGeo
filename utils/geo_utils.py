from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from retrying import retry
import requests
import json
from utils.my_utils import *

def Mydistance(points1, points2):
    """
    points1, points2 should be a list:
        [(lat, lon), ...]
    """
    min_dis = 999999
    for point1 in points1:
        for point2 in points2:
            dis = geodesic(point1, point2).kilometers
            if dis < min_dis:
                min_dis = dis
    return min_dis

@retry(stop_max_attempt_number=3, wait_fixed=10000)
def get_location_general(text, TOKEN=None):
    """
    TOKEN: None -- osm query
    """
    place = text
    res = {'has_answer': False}
    res['city'] = "None"
    res['country'] = "None"
    if TOKEN == None:
        geolocator = Nominatim(user_agent="lonlat_locator")
        try:
            location = geolocator.geocode(place)
        except Exception as e:
            # print(e)
            location = None
        if location == None:
            return res
        else:
            res['has_answer'] = True
            res['points'] = [(location.latitude, location.longitude)]
            return res
    else:
        url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={place}&lang=en&limit=1&format=json&apiKey={TOKEN}"
        response = requests.get(url)
        res_dict = response.json()
        if len(res_dict['results']) == 0:
            return res
        else:
            res['has_answer'] = True
            keys_list = res_dict['results'][0].keys()
            if 'bbox' in keys_list:
                res['points'] = [(res_dict['results'][0]['bbox']['lat1'], res_dict['results'][0]['bbox']['lon1']), (res_dict['results'][0]['bbox']['lat2'], res_dict['results'][0]['bbox']['lon2'])]
            else:
                # set a non-country area
                res['points'] = [(90,0)]
            if 'city' in keys_list:
                res['city'] = res_dict['results'][0]['city']
            else:
                res['city'] = 'None'
            if 'country' in keys_list:
                res['country'] = res_dict['results'][0]['country']
            else:
                res['country'] = 'None'
            return res

@retry(stop_max_attempt_number=3, wait_fixed=10000)
def get_location_str(city, country, TOKEN=None):
    """
    TOKEN: None -- osm query
    """
    city = city.lower()
    city = extract_after_substring(city, 'located in ')
    city = extract_after_substring(city, 'located on ')
    city = extract_after_substring(city, 'located at ')
    city = extract_after_substring(city, 'in the region of ')
    city = extract_after_substring(city, 'location in ')
    city = extract_after_substring(city, 'characteristic of ')
    city = extract_after_substring(city, 'possibly in ')
    city = extract_after_substring(city, 'is ')
    city = extract_after_substring(city, 'is of ')
    country = country.lower()
    country = extract_after_substring(country, 'located in ')
    country = extract_after_substring(country, 'located on ')
    country = extract_after_substring(country, 'located at ')
    country = extract_after_substring(country, 'in the region of ')
    country = extract_after_substring(country, 'location in ')
    country = extract_after_substring(country, 'characteristic of ')
    country = extract_after_substring(country, 'possibly in ')
    country = extract_after_substring(country, 'is ')
    country = extract_after_substring(country, 'is of ')
    place = city + ', ' + country
    res = {'has_answer': False}
    res['city'] = city
    res['country'] = country
    if ('<city>' in city) or ('None' in city) or ('' == city) or ('<country>' in country) or ('None' in country) or ('' == country):
        return res
    if TOKEN == None:
        geolocator = Nominatim(user_agent="lonlat_locator")
        try:
            location = geolocator.geocode(place)
        except Exception as e:
            # print(e)
            location = None
        if location == None:
            return res
        else:
            res['has_answer'] = True
            res['points'] = [(location.latitude, location.longitude)]
            return res
    else:
        url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={place}&lang=en&limit=1&format=json&apiKey={TOKEN}"
        response = requests.get(url)
        res_dict = response.json()
        if len(res_dict['results']) == 0:
            return res
        else:
            res['has_answer'] = True
            keys_list = res_dict['results'][0].keys()
            if 'bbox' in keys_list:
                res['points'] = [(res_dict['results'][0]['bbox']['lat1'], res_dict['results'][0]['bbox']['lon1']), (res_dict['results'][0]['bbox']['lat2'], res_dict['results'][0]['bbox']['lon2'])]
            else:
                # set a non-country area
                res['points'] = [(90,0)]
            if 'city' in keys_list:
                res['city'] = res_dict['results'][0]['city']
            else:
                res['city'] = 'None'
            if 'country' in keys_list:
                res['country'] = res_dict['results'][0]['country']
            else:
                res['country'] = 'None'
            return res

def are_same_place(res1, res2, dis_th=100):
    if res1['has_answer'] and res2['has_answer']:
        distance = Mydistance(points1=res1['points'], points2=res2['points'])
        return distance < dis_th
    elif ((res1['city'] in res2['city']) or (res2['city'] in res1['city'])) and ((res1['country'] in res2['country']) or (res2['country'] in res1['country'])):
        return True
    else:
        return False