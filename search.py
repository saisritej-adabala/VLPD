import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["license"]
mycol = mydb["vehicalsdata"]
plate=input("ENTER THE VEHICLE NUMBER TO BE SEARCHED: ")
print()
print("Vehicles Found : ")
print()
print()
for x in mycol.find({"Numberplate":plate},{ "_id": 0}):
    print("NumberPlate : "+x['Numberplate'])
    print("Time        : "+x['Rtime'])
    print("Location    : "+x['Location'])
    print()

