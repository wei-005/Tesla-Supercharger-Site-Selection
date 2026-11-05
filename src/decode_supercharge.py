"""Decode the packed supercharge.info export (CONUS sites) into a CSV.
Row layout (17 chars): lat(3,b36,(lat-24)*1000) lng(4,b36,(lng+125)*1000) date(3,b36 days since 2010-01-01 or 'zzz')
stalls(2) kW/5(2) status(1) flags(1: otherEVs*2+battery) facility_category(1)"""
import sys, datetime as dt, pandas as pd
ST={'O':'OPEN','C':'CONSTRUCTION','P':'PERMIT','L':'PLAN','E':'EXPANDING','X':'CLOSED_PERM','T':'CLOSED_TEMP','V':'VOTING'}
CAT={'F':'fuel_travel_center','G':'grocery_bigbox','M':'mall_shopping_center','H':'hotel_casino','R':'restaurant','Z':'other','N':'unknown'}
p=''.join(open(f).read().strip() for f in sys.argv[1:-1])
rows=[]
for i in range(0,len(p),17):
    r=p[i:i+17]
    lat=int(r[0:3],36)/1000+24; lng=int(r[3:7],36)/1000-125
    d=None if r[7:10]=='zzz' else (dt.date(2010,1,1)+dt.timedelta(days=int(r[7:10],36))).isoformat()
    fl=int(r[15])
    rows.append(dict(lat=lat,lng=lng,date_opened=d,stalls=int(r[10:12],36),power_kw=int(r[12:14],36)*5,
                     status=ST[r[14]],other_evs=bool(fl&2),battery=bool(fl&1),facility=CAT[r[16]]))
df=pd.DataFrame(rows); df.to_csv(sys.argv[-1],index=False)
print(df.shape); print(df.status.value_counts()); print(df.facility.value_counts())
print(pd.to_datetime(df.date_opened).dt.year.value_counts().sort_index())
