import sys
import re
import operator

interval = 0.01

def getDeviceIdFromExecution(s):
    words = s.split()
    if len(words) < 4 or words[3] != 'execute':
        return None
    else:
        return int(words[2][1:])

def getDeviceIdFromCreation(s):
    words = s.split()
    if len(words) < 6 or words[0] != 'create' or words[1] != 'new' or words[2] != 'op':
        return None
    else:
        return int(words[7][1:])

def getDeviceIdFromDeletion(s):
    words = s.split()
    if len(words) < 6 or words[0] != 'dispatcher' or words[1] != 'ready':
        return None
    else:
        return 0

def parseSecond(s):
    hour, minute, second = s.split(':')
    return int(hour) * 3600 + int(minute) * 60 + float(second)

def parseFile(filename, deviceIdParser):
    ret = {}
    with open(filename) as f:
        for line in f.readlines():
            words = line.split(None, 4)
            time = parseSecond(words[1])
            device = deviceIdParser(words[4])
            if device == None:
                continue
            else:
                bucket = int(time / interval)
                ret.setdefault(bucket, dict())
                ret[bucket].setdefault(device, 0)
                ret[bucket][device] += 1
    return ret

def outputBuckets(l):
    union = list(set.union(*map(set, map(dict.keys, l))))
    timeAxis = range(min(union), max(union) + 1)
    deviceIds = set.union(*map(set, map(dict.keys, reduce(operator.add, map(dict.values, l)))))
    with open(sys.argv[1] + '.hist', 'w') as f:
        f.write(','.join(['time'] + reduce(operator.add, [[str(bucketId) + '_device' + str(deviceId) for deviceId in deviceIds] for bucketId in range(len(l))])) + '\n')
        for time in timeAxis:
            f.write(','.join(map(str, [time] + reduce(operator.add, [[l[bucketId].get(time, dict()).get(deviceId, 0) for deviceId in deviceIds] for bucketId in range(len(l))]))) + '\n')


def main():
    assert(1 < len(sys.argv))
    filename = sys.argv[1]
    execution = parseFile(filename, getDeviceIdFromExecution)
    creation = parseFile(filename, getDeviceIdFromCreation)
    deletion = parseFile(filename, getDeviceIdFromDeletion)
    outputBuckets([execution, creation, deletion])

if __name__ == '__main__':
    main()
