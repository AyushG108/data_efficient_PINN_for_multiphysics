def load_sparse_temp_csv(fn):
    df = pd.read_csv(fn)
    names = [c.lower() for c in df.columns]
    x = df.columns[names.index('x')]
    y = df.columns[names.index('y')]
    t = df.columns[names.index('t')] if 't' in names else df.columns[-1]
    return df[[x,y,t]].values.astype(np.float32)