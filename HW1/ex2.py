import argparse
import time
import psutil as psu
import redis
import uuid

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)

    args = parser.parse_args()

    redis_cl = create_redis_connection(args)

    # Retention Periods for 5MB and 1MB constraints in milliseconds
    rtp_5mb = (3276800 * 1) * 1000
    rtp_1mb = (655360 * (24 * 60 * 60) ) * 1000

    # Time Series naming
    ts1 = get_mac_addr() + ':battery'
    ts2 = get_mac_addr() + ':power'
    ts3 = get_mac_addr() + ':plugged_seconds'
    ts_dict = {ts1:rtp_5mb, ts2:rtp_5mb, ts3:rtp_1mb}

    for ts, retention_period in ts_dict.items():
        print()
        create_ts(redis_cl, ts, retention_period, 128)

    process_timeseries(redis_cl, ts1, ts2, ts3) 


def process_timeseries(redis_client, ts1, ts2, ts3):

    print("\nRecording in progress ...")
    print()

    one_day_s = 60 * 60 * 24
    prev_time_s = time.time()
    prev_time_d = time.time()
    plugged_seconds = 0

    while(True):

        # Delta 1 second passed
        delta_sec = time.time() - prev_time_s
        # Delta 24 hours passed
        delta_day = time.time() - prev_time_d

        timestamp_ms = int(time.time() * 1000)

        if delta_sec > 1:

            battery = psu.sensors_battery()

            if battery.power_plugged == 1:
                plugged_seconds = plugged_seconds + 1

            redis_client.ts().add(ts1, timestamp_ms, battery.percent)
            redis_client.ts().add(ts2, timestamp_ms, int(battery.power_plugged))

            # if 1 second passed, reset prev_time_s
            prev_time_s = time.time()

        if delta_day > one_day_s:
            redis_client.ts().add(ts3, timestamp_ms, plugged_seconds)
            # if 24 hours passed, reset prev_time_d
            prev_time_d = time.time()
            plugged_seconds = 0


def create_redis_connection(args):

    print('Creating connection to Redis ...')

    redis_cl = redis.Redis(host=args.host, port=args.port, username=args.user, password=args.password)
    is_connected = redis_cl.ping()

    print('Redis Connected: ', is_connected)

    return redis_cl


def get_mac_addr():
    mac = hex(uuid.getnode())
    return mac


def create_ts(redis_client, ts_name, retention_period, chnk_size=128):
    
    try:
        print('Creating timeseries %s' % (ts_name))

        if(redis_client.exists(ts_name) == 0):
            redis_client.ts().create(
                ts_name, 
                retention_msecs=retention_period,
                uncompressed=False,
                chunk_size=chnk_size
            )
            print(f'Timeseries {ts_name} created!')

        else:

            print('Time series %s already exists.' % (ts_name))

            lastSample = redis_client.ts().get(ts_name)

            if(lastSample is None):
                 print('Time series is empty. No preceding records found.')
            else:
                if(int(time.time()*1000) - lastSample[0] > retention_period):
                    print('WARNING: last time series added (%s) is older than retention period. Not deleting preceding records will cause an error! For more info: https://redis.io/commands/ts.add/' % (str(lastSample)))

                print('Do you want to delete preceding records? [Y/N]')
                
                while(True):
                    
                    key = input()
                    if key in ('y', 'Y'):
                    
                        timestamp_ms_to = int(time.time() * 1000)
                        timestamp_ms_from = timestamp_ms_to - int(time.time() * 1000)
                        redis_client.ts().delete(
                            ts_name, 
                            from_time=timestamp_ms_from, 
                            to_time=timestamp_ms_to
                        )
                        print('All preceding data were deleted!')
                        break
                    elif key in ('n', 'N'):
                        break
                    else:
                        print('Sorry, only possible input: [Y/N]')

    except redis.ResponseError:
        print(f'An error occured while creating the timeseries {ts_name}.')
        pass


if __name__ == '__main__':
    main()