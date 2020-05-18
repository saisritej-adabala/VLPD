import reverse_geocoder as rg 
  
def reverseGeocode(coordinates): 
    result = rg.search(coordinates) 
    return result[0]['name']+", "+result[0]['admin2']+", "+result[0]['admin1']

if __name__=="__main__": 
 
    coordinates =(17.802280, 83.385150) 
    res=reverseGeocode(coordinates)
    print(res)

