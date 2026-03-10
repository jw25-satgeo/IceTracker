import math

metric2base = {
    'W':    ('W', 1.0),
    'kW':   ('W', 1e3),
    'dB':   ('dB', 1.0),     # keep dimensionless
    'dBHz': ('dBHz', 1.0),   # also dimensionless (log domain)
    
    # Doppler rate, chirp rate, FM rate
    'Hz/s':    ('Hz/s', 1.0),
    'kHz/s':   ('Hz/s', 1e3),
    'Hz/ms':   ('Hz/s', 1e3),
    # PRF sometimes shows up as 1/s
    '1/s':     ('Hz', 1.0),

    'rad':   ('rad', 1.0),
    'deg':   ('rad', math.pi / 180.0),
    'mrad':  ('rad', 1e-3),
    
     # Length
    'm':    ('m', 1.0),
    'km':   ('m', 1e3),
    'cm':   ('m', 1e-2),
    'mm':   ('m', 1e-3),
    'µm':   ('m', 1e-6),
    'nm':   ('m', 1e-9),
    'm/s':   ('m/s', 1.0),
    'km/s':  ('m/s', 1e3),
    'km/h':  ('m/s', 1e3 / 3600),

    # Frequency
    'Hz':   ('Hz', 1.0),
    'kHz':  ('Hz', 1e3),
    'MHz':  ('Hz', 1e6),
    'GHz':  ('Hz', 1e9),

    # Time
    's':    ('s', 1.0),
    'ms':   ('s', 1e-3),
    'us':   ('s', 1e-6),
    'µs':   ('s', 1e-6),
    'ns':   ('s', 1e-9),
}