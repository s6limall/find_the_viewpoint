import requests
import time

for i in range(4,131):
    url = 'http://neilsloane.com/packings/dim3/'
    name = 'pack.3.' + str(i) + '.txt'
    r = requests.get(url + name)
    with open('Sphere_Codes/' + name, "wb") as code:
        code.write(r.content)
        #time.sleep(5)
