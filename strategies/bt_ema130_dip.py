"""Backtest de btc_ema130_dip.pine (BTC 1D spot, long/cash).
Usage : python3 bt_ema130_dip.py btc.csv   (colonnes t,o,h,l,c ; t = date ou timestamp unix)
Signal à la clôture, exécution à l'ouverture suivante, 0,15 %/côté.
Short (perpétuel) : financement estimé à 10 %/an en moyenne."""
import sys, numpy as np, pandas as pd

df = pd.read_csv(sys.argv[1])
df['t'] = pd.to_datetime(df.t, unit='s') if pd.api.types.is_numeric_dtype(df.t) else pd.to_datetime(df.t)
df = df.set_index('t')
O, C = df.o, df.c
FEE, START, FUND = 0.0015, '2020-10-01', 0.10
ema = lambda x, n: x.ewm(span=n, adjust=False).mean()

def signal(emaLen=130, exitBuf=0.03, dip=True, depth=0.25, exLen=20, dmin=5, dmax=40, stop=0.15, cool=20,
           short=False, shortSize=1.0, bearLen=365, bearLook=10):
    sma = C.rolling(bearLen).mean(); bear = (sma < sma.shift(bearLook)) & (C < sma)
    e = ema(C, emaLen); ex = ema(C, exLen); up = e > e.shift(); dn = e < e.shift()
    pos = np.zeros(len(C)); trend = 0; on = False; k = 0; px = 0.0; last = -10**9
    for i in range(len(C)):
        if up.iat[i]: trend = 1
        elif dn.iat[i] and C.iat[i] < e.iat[i] * (1 - exitBuf): trend = 0
        if trend == 1: on = False
        elif not on:
            if dip and C.iat[i] < e.iat[i] * (1 - depth) and i - last > cool:
                on, k, px = True, 0, C.iat[i]
        else:
            k += 1
            if (C.iat[i] > ex.iat[i] and k >= dmin) or k >= dmax or C.iat[i] < px * (1 - stop):
                on, last = False, i
        pos[i] = 1 if (trend == 1 or on) else (-shortSize if short and bear.iat[i] else 0)
    return pd.Series(pos, index=C.index)

def bt(sig, start=START, end=None):
    s = sig.shift(1).fillna(0); r = (O.shift(-1) / O - 1).fillna(0)
    m = (df.index >= start) & ((df.index <= end) if end else True)
    s, r = s[m], r[m]; turn = s.diff().abs().fillna(abs(s.iloc[0]))
    eq = ((1 + s * r - (s < 0) * (-s) * FUND / 365) * (1 - turn * FEE)).cumprod(); dr = eq.pct_change().fillna(0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(total=f"{(eq.iloc[-1]-1)*100:.0f}%", cagr=f"{(eq.iloc[-1]**(1/yrs)-1)*100:.1f}%",
                dd=f"{(eq/eq.cummax()-1).min()*100:.1f}%", sharpe=round(dr.mean()/dr.std()*np.sqrt(365), 2),
                ordres=int((turn > 0).sum()))

if __name__ == '__main__':
    df = df.iloc[:-1]; O, C = df.o, df.c          # dernière bougie encore en formation
    for nom, kw in [('ancienne (EMA130 seule)', dict(exitBuf=0, dip=False)), ('long/cash + krachs', {}), ('+ short marché baissier', dict(short=True))]:
        s = signal(**kw)
        print(nom, bt(s), '| 2018-11→2020-09 :', bt(s, '2018-11-01', '2020-09-30'))
