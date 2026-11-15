//! Native cores for `src/ytm_player/services/` — the backend service engines.

pub(crate) mod auth;
pub(crate) mod cache;
pub(crate) mod discord_rpc;
pub(crate) mod download;
pub(crate) mod history;
pub(crate) mod lastfm;
pub(crate) mod lrclib;
pub(crate) mod platform_media;
pub(crate) mod player;
pub(crate) mod queue;
pub(crate) mod regions;
pub(crate) mod shuffle_prefs;
pub(crate) mod spotify_import;
pub(crate) mod stream;
pub(crate) mod update_check;
pub(crate) mod yt_dlp_options;
pub(crate) mod ytmusic;
