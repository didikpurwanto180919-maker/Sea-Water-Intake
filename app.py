@st.cache_resource
def train_jellyfish_model():
  np.random.seed(42)
  n_samples = 500  # Dikurangi dari 1500 agar proses training instant

  sst = np.random.normal(loc=29.5, scale=1.2, size=n_samples)
  salinity = np.random.normal(loc=32.5, scale=1.1, size=n_samples)
  current_speed = np.random.exponential(scale=0.25, size=n_samples)
  chlorophyll = np.random.gamma(shape=2.2, scale=0.7, size=n_samples)
  wind_speed = np.random.uniform(1.0, 12.0, size=n_samples)

  risk_score = (
      (sst - 28.5) * 0.35
      + (chlorophyll) * 0.35
      + (current_speed) * 0.15
      + (wind_speed * 0.15)
  )

  labels = pd.qcut(risk_score, q=3, labels=[0, 1, 2])

  df = pd.DataFrame({
      "sst": sst,
      "salinity": salinity,
      "current_speed": current_speed,
      "chlorophyll_a": chlorophyll,
      "wind_speed": wind_speed,
      "risk_level": labels,
  })

  X = df.drop(columns=["risk_level"])
  y = df["risk_level"]

  model = XGBClassifier(
      n_estimators=30,  # Dikurangi dari 100 ke 30 (Sangat cepat)
      learning_rate=0.05,
      max_depth=3,
      eval_metric="mlogloss",
  )
  model.fit(X, y)
  return model
