from subprocess import call
def locs():
    f = open("loc.txt", "w+")
    x=call(["python","location.py"],stdout=f)
    fw=open("loc.txt","r+")
    lines=fw.readlines()
    s=str(lines[1]).strip()
    return s
