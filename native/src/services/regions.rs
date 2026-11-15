//! Native cores for `src/ytm_player/services/regions.py` — the chart
//! region table and the locale normaliser.

use pyo3::prelude::*;

/// The chart-region table behind the region picker, the charts queries
/// and the shelf-title country suffix strip — the `CHART_REGIONS`
/// constant of `ytm_player.services.regions`.
///
/// The 68 entries in the reference's exact order: `ZZ` (Global) first
/// as the default, then alphabetical by display name — the 62 codes
/// YouTube's `get_charts` `countries.options` advertises plus the six
/// historically-supported extras (Hong Kong, Malaysia, Singapore,
/// Taiwan, Thailand, Vietnam). The engine answers the pairs as a list
/// of `(code, name)` string tuples; the module wraps it into the
/// tuple-of-tuples shape the constant spells.
#[pyfunction]
pub(crate) fn chart_regions() -> Vec<(String, String)> {
    const REGIONS: &[(&str, &str)] = &[
        ("ZZ", "Global"),
        ("AR", "Argentina"),
        ("AU", "Australia"),
        ("AT", "Austria"),
        ("BE", "Belgium"),
        ("BO", "Bolivia"),
        ("BR", "Brazil"),
        ("CA", "Canada"),
        ("CL", "Chile"),
        ("CO", "Colombia"),
        ("CR", "Costa Rica"),
        ("CZ", "Czechia"),
        ("DK", "Denmark"),
        ("DO", "Dominican Republic"),
        ("EC", "Ecuador"),
        ("EG", "Egypt"),
        ("SV", "El Salvador"),
        ("EE", "Estonia"),
        ("FI", "Finland"),
        ("FR", "France"),
        ("DE", "Germany"),
        ("GT", "Guatemala"),
        ("HN", "Honduras"),
        ("HK", "Hong Kong"),
        ("HU", "Hungary"),
        ("IS", "Iceland"),
        ("IN", "India"),
        ("ID", "Indonesia"),
        ("IE", "Ireland"),
        ("IL", "Israel"),
        ("IT", "Italy"),
        ("JP", "Japan"),
        ("KE", "Kenya"),
        ("LU", "Luxembourg"),
        ("MY", "Malaysia"),
        ("MX", "Mexico"),
        ("NL", "Netherlands"),
        ("NZ", "New Zealand"),
        ("NI", "Nicaragua"),
        ("NG", "Nigeria"),
        ("NO", "Norway"),
        ("PA", "Panama"),
        ("PY", "Paraguay"),
        ("PE", "Peru"),
        ("PL", "Poland"),
        ("PT", "Portugal"),
        ("RO", "Romania"),
        ("RU", "Russia"),
        ("SA", "Saudi Arabia"),
        ("RS", "Serbia"),
        ("SG", "Singapore"),
        ("ZA", "South Africa"),
        ("KR", "South Korea"),
        ("ES", "Spain"),
        ("SE", "Sweden"),
        ("CH", "Switzerland"),
        ("TW", "Taiwan"),
        ("TZ", "Tanzania"),
        ("TH", "Thailand"),
        ("TR", "Turkey"),
        ("UG", "Uganda"),
        ("UA", "Ukraine"),
        ("AE", "United Arab Emirates"),
        ("GB", "United Kingdom"),
        ("US", "United States"),
        ("UY", "Uruguay"),
        ("VN", "Vietnam"),
        ("ZW", "Zimbabwe"),
    ];
    REGIONS
        .iter()
        .map(|(code, name)| (code.to_string(), name.to_string()))
        .collect()
}

/// Normalise a YouTube chart region value to an ISO 3166-1 alpha-2 code.
/// Mirrors `ytm_player.services.regions.normalise_region`.
///
/// Accepts a bare code ("ES", "es") or a language-territory locale
/// ("en-GB", "es_ES", " EN-gb ") and returns the territory uppercased.
/// Non-locale input is returned uppercased as-is; empty input stays empty.
#[pyfunction]
pub(crate) fn normalise_region(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }

    let replaced = value.trim().replace('_', "-");

    // Partition on the first '-', mirroring str.partition("-").
    let (head, territory) = match replaced.split_once('-') {
        Some((h, t)) => (h, Some(t)),
        None => (replaced.as_str(), None),
    };

    if let Some(t) = territory {
        if head.len() >= 2
            && head.len() <= 3
            && head.is_ascii()
            && head.chars().all(char::is_alphabetic)
            && t.len() == 2
            && t.is_ascii()
            && t.chars().all(char::is_alphabetic)
        {
            return t.to_uppercase();
        }
    }

    head.trim().to_uppercase()
}
