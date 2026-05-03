# CHANGELOG

<!-- version list -->

## v1.3.0 (2026-05-03)

### Features

- **dashboard**: UI polish — overview icons/styles, dynamic footer version, repeat OD chart
  simplification
  ([`8f210bf`](https://github.com/stompsy/cpmdash/commit/8f210bf0dddb3d4ae7f47769be41c1164b577ece))


## v1.2.0 (2026-04-30)

### Features

- **dashboard**: Expand Hargrove reporting and overdose analytics
  ([`916e17b`](https://github.com/stompsy/cpmdash/commit/916e17b5738911d9d024964dd897d0764bbb3c4f))


## v1.1.1 (2026-04-22)

### Bug Fixes

- **dashboard**: Polish co-occurring deep-dive layout and printable view
  ([#8](https://github.com/stompsy/cpmdash/pull/8),
  [`00649c3`](https://github.com/stompsy/cpmdash/commit/00649c3ced0388536e27ea19d12af2c20145d6fc))


## v1.1.0 (2026-04-20)

### Features

- Add co-occurring disorders deep-dive page with donut center annotations
  ([#7](https://github.com/stompsy/cpmdash/pull/7),
  [`06fc05f`](https://github.com/stompsy/cpmdash/commit/06fc05faf258f847c2d66fb4bb6f1b12a3d211b3))


## v1.0.3 (2026-04-19)

### Bug Fixes

- Strip timezone before to_period() in OD monthly histogram
  ([`1bc4dbc`](https://github.com/stompsy/cpmdash/commit/1bc4dbc38f534f45ce4825316732e4b3f4208f2e))


## v1.0.2 (2026-04-19)

### Bug Fixes

- Pin non-geocodable addresses to Serenity House and skip API calls
  ([`592898b`](https://github.com/stompsy/cpmdash/commit/592898bd6a302381c8a9e4f243182146e3c70079))


## v1.0.1 (2026-04-19)

### Bug Fixes

- Switch to gthread workers to fix SSE heartbeat during geocoding
  ([`416cbd5`](https://github.com/stompsy/cpmdash/commit/416cbd531d62a746b4e149e72257da4b7f013426))


## v1.0.0 (2026-04-19)

### Bug Fixes

- Add missing migration for row_status choices (adds 'changed')
  ([`fdd0b84`](https://github.com/stompsy/cpmdash/commit/fdd0b84aac44291da71d43215a3b0b3048fa8371))

- Correct semantic-release branches config for v9+
  ([`b2c0888`](https://github.com/stompsy/cpmdash/commit/b2c08886ccd6ee3b5b0e5de3352e9c70515f9370))

- Dockerfile only copies assets/css/ for Tailwind build, not raw data CSVs
  ([`21f64a7`](https://github.com/stompsy/cpmdash/commit/21f64a713ccf8029fbe48be09dac87f08bf3dd94))

- Include contourpy in requirements for Railway deploy
  ([`874eed0`](https://github.com/stompsy/cpmdash/commit/874eed07ed19eb222a27a3379f1096bdbb5ce254))

- Prevent Gunicorn worker timeout during SSE geocoding
  ([#6](https://github.com/stompsy/cpmdash/pull/6),
  [`94670e0`](https://github.com/stompsy/cpmdash/commit/94670e0ad5e8c7b721d49453c0e239308e7c5b27))

- Prevent validation warnings from stealing NEW status in row classification
  ([#3](https://github.com/stompsy/cpmdash/pull/3),
  [`2b7b924`](https://github.com/stompsy/cpmdash/commit/2b7b9242805bc6e59c34589f3a78281208d87922))

- Remove build_command from semantic-release config
  ([`bc96b71`](https://github.com/stompsy/cpmdash/commit/bc96b71417c34955821d172de35304acba1898eb))

- Remove COPY of gitignored staticfiles/ and media/ from Dockerfile
  ([`f14379d`](https://github.com/stompsy/cpmdash/commit/f14379df5169100acba940943f17ddbfe1356e69))

- Restore dynamic 2025 data for Hargrove report and update release config
  ([`100b1e8`](https://github.com/stompsy/cpmdash/commit/100b1e86368edb23af20f5d0ef6ed553d5cb5b8f))

- Restore dynamic 2025 data for Hargrove report and update release config
  ([`520d2e4`](https://github.com/stompsy/cpmdash/commit/520d2e49976f7593563f39aab47e9f75217a0eb5))

- Track assets/css/input.css for Docker Tailwind build
  ([`5d53c4c`](https://github.com/stompsy/cpmdash/commit/5d53c4cda6ad119ab4416fc22ce4e4e03d07bff7))

- Upgrade Django 5.2.8, refactor od_map template, fix test auth compatibility
  ([`257b460`](https://github.com/stompsy/cpmdash/commit/257b460b172c7bab97ae7c5a589670b627dcce71))

- **ci**: Use 'version' instead of 'publish' in release workflow
  ([`df18452`](https://github.com/stompsy/cpmdash/commit/df1845215e4d155e6823190cab4bdc0a7da8d060))

- **config**: Sanitize CSRF_TRUSTED_ORIGINS parsing
  ([`304134e`](https://github.com/stompsy/cpmdash/commit/304134ed42a88adf5de979e9fffef158a2902b55))

- **dashboard**: Show repeat-OD quick stat values
  ([`a8b1e7d`](https://github.com/stompsy/cpmdash/commit/a8b1e7def23a0af02a292f1580ecccab88234909))

- **overdose**: Compute repeat OD quick stats reliably
  ([`fee92d7`](https://github.com/stompsy/cpmdash/commit/fee92d7d6a82cc1cd0d34a242a3bc9ef7cf22905))

- **release**: Checkout branch name instead of SHA to avoid detached HEAD
  ([#4](https://github.com/stompsy/cpmdash/pull/4),
  [`14d5a60`](https://github.com/stompsy/cpmdash/commit/14d5a608b60c6b905d2a9446ea99557cfad9ae50))

- **release**: Restore semantic-release version baseline
  ([`b45dfa7`](https://github.com/stompsy/cpmdash/commit/b45dfa765a1c0c9744465fed007b73e77576caf4))

- **ui**: Pause signups and refresh timeline/contact
  ([`5f03a96`](https://github.com/stompsy/cpmdash/commit/5f03a96c12852c46c400f33d68650e18a5235210))

### Chores

- Clean up code from removed quick stats
  ([`2aa7a72`](https://github.com/stompsy/cpmdash/commit/2aa7a72e9288dc3c211a4cb5e76e987b8586c48e))

- Remove unused Dockerfile and .dockerignore
  ([`6d070ee`](https://github.com/stompsy/cpmdash/commit/6d070eeb3fcf2d768225664528d084fcc0cd9150))

### Continuous Integration

- Remove Docker build job
  ([`c14ff7b`](https://github.com/stompsy/cpmdash/commit/c14ff7b372ce69b4427b7031a7e332e49f7592eb))

### Features

- Add expand-to-modal controls across dashboard charts
  ([`46b2b5a`](https://github.com/stompsy/cpmdash/commit/46b2b5ad81a46dda80c351c20c181b19e3c68e6c))

- Add patient address map with KDE density contour overlay
  ([`d85daef`](https://github.com/stompsy/cpmdash/commit/d85daef04bfe61017080cc17a81aae1298c3b514))

- Add Timeline app and Hargrove Grant dashboard, disable search
  ([`26c57c0`](https://github.com/stompsy/cpmdash/commit/26c57c007cf07da7491bd4a68e0741914847f299))

- Address-level geocoding, expanded import pipeline, and data fixes
  ([#5](https://github.com/stompsy/cpmdash/pull/5),
  [`3af3c28`](https://github.com/stompsy/cpmdash/commit/3af3c288c082ebc5a3aaf6284beaadf6da695624))

- Auth lockdown, OD referrals expansion, dashboard polish
  ([`38b419e`](https://github.com/stompsy/cpmdash/commit/38b419e6190774f7c0278b4d1a4abf1350227bb7))

- Data Import ETL Pipeline with Field Cleaning and Geocoding
  ([#2](https://github.com/stompsy/cpmdash/pull/2),
  [`e8285d9`](https://github.com/stompsy/cpmdash/commit/e8285d9c6c720cf27dd86693a8ad8b8103497964))

- Data Import Management Console ([#1](https://github.com/stompsy/cpmdash/pull/1),
  [`454436e`](https://github.com/stompsy/cpmdash/commit/454436ece11301f4d12a6ef5599fc59dd70cb0dd))

- **charts**: Update age referral Sankey flow counts to include missing categories
  ([`9d0e06b`](https://github.com/stompsy/cpmdash/commit/9d0e06bfa89b147de4149a74af9614187ffb93ba))

- **dashboard**: Enhance SUD referral and responder analysis
  ([`7173e36`](https://github.com/stompsy/cpmdash/commit/7173e3676b5de3fdf3a646354fac3ae72f3e1708))

- **dashboard**: Expand repeat OD insights and missing-data toggles
  ([`154fa0e`](https://github.com/stompsy/cpmdash/commit/154fa0e6cfde4ba6df3cb95e42d6d0bd1c83521c))

- **home**: Add services provided section
  ([`53010af`](https://github.com/stompsy/cpmdash/commit/53010af5e84e2f7ba92261b0c32ddff2b1407aba))

- **ui**: Refine timeline connectors, auth navigation, and cost analysis content
  ([`71e1d20`](https://github.com/stompsy/cpmdash/commit/71e1d20e700cf632f6774d1597babbb96d5541e9))

### Refactoring

- **ui**: Standardize page layouts and redesign contact submissions
  ([`41a669f`](https://github.com/stompsy/cpmdash/commit/41a669fb181ad125fd88b20c89ece659f4221a7b))


## v0.1.0 (2025-10-23)

- Initial Release
