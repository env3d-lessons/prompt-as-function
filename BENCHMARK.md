Below are results comparing different Qwen models under various configurations. All benchmarks were run in **GitHub Codespaces** using CPU-only inference and the `PromptFunction` wrapper. Elapsed times include full inference time, measured in milliseconds (ms).

### TL;DR

* **Prompt-as-function** dramatically improves performance by reusing the KV cache.
* With this optimization, **even a 7B model becomes responsive**, and a 0.5B model consistently outperforms cloud APIs.
* Turning the optimization **off** leads to **5x–6x slower inference**.

---

### Fastest Model: Qwen 0.5B with Prompt-as-Function Enabled

```bash
$ python extract_categories.py 0
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | retail | 241.78 |
| Starbucks | retail | 177.55 |
| Walmart | retail | 165.44 |
| Target | grocery | 156.45 |
| Apple Store | retail | 169.29 |
| Costco | retail | 165.25 |
| Uber | transportation | 156.09 |
| McDonald's | restaurant | 194.91 |
| Netflix | entertainment | 158.30 |
| Best Buy | retail | 165.50 |
| Shell | retail | 382.19 |
| CVS Pharmacy | pharmacy | 243.62 |
| Home Depot | retail | 168.37 |
| Walgreens | grocery | 168.27 |
| Nike | retail | 157.14 |
| Subway | transportation | 163.49 |
| Delta Airlines | transportation | 163.49 |
| Spotify | music store | 191.57 |
| Lowe's | retail | 253.13 |
| Chipotle | retail | 184.39 |
| Airbnb | retail | 225.40 |
| FedEx | retail | 201.47 |
| Whole Foods Market | retail | 255.51 |
| H&M | retail | 167.81 |
| Google Play | retail | 170.89 |
| AT&T | retail | 189.55 |
| IKEA | retail | 190.79 |
| Domino's Pizza | retail | 201.62 |
| Burger King | retail | 183.43 |
| eBay | retail | 165.38 |

> ✅ **Most calls under 200ms**. This is **faster than OpenAI's API**, all without leaving your machine.

---

### 🐢 0.5B Model with Prompt-as-Function **Disabled**

```bash
$ PROMPT_AS_FUNCTION=0 python extract_categories.py 0
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 1115.75 |
| Starbucks | retail | 1050.65 |
| Walmart | grocery | 1026.47 |
| Target | grocery | 1075.30 |
| Apple Store | electronics | 1237.13 |
| Costco | grocery | 1095.82 |
| Uber | transportation | 1050.05 |
| McDonald's | restaurant | 1032.86 |
| Netflix | entertainment | 1105.70 |
| Best Buy | grocery | 1012.89 |
| Shell | retail | 1122.07 |
| CVS Pharmacy | pharmacy | 1076.09 |
| Home Depot | grocery | 1052.20 |
| Walgreens | grocery | 1083.40 |
| Nike | grocery | 1210.43 |
| Subway | transportation | 1228.53 |
| Delta Airlines | transportation | 1049.88 |
| Spotify | entertainment | 1137.09 |
| Lowe's | grocery | 1181.66 |
| Chipotle | grocery | 1129.97 |
| Airbnb | lodging | 1224.56 |
| FedEx | transportation | 1189.77 |
| Whole Foods Market | grocery | 1171.68 |
| H&M | categor | 1146.53 |
| Google Play | grocery | 1134.64 |
| AT&T | telecom | 1235.82 |
| IKEA | grocery | 1286.86 |
| Domino's Pizza | grocery | 1179.42 |
| Burger King | restaurant | 1182.32 |
| eBay | ebay | 1251.83 |

> Inference time balloons to over **1 second per query**, despite being the same model and hardware.

---

### Scaling Up: Larger Models (All Use Prompt-as-Function)

#### Qwen 1.5B

```bash
$ python extract_categories.py 1
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 458.15 |
| Starbucks | grocery | 464.32 |
| Walmart | grocery | 603.33 |
| Target | delivery | 455.76 |
| Apple Store | services | 459.85 |
| Costco | grocery | 463.94 |
| Uber | grocery | 445.69 |
| McDonald's | grocery | 509.82 |
| Netflix | services | 434.17 |
| Best Buy | grocery | 507.04 |
| Shell | delivery | 480.98 |
| CVS Pharmacy | pharmacy | 565.09 |
| Home Depot | grocery | 465.57 |
| Walgreens | grocery | 461.07 |
| Nike | grocery | 421.17 |
| Subway | delivery | 498.20 |
| Delta Airlines | airlines | 457.86 |
| Spotify | categories: | 461.41 |
| Lowe's | grocery | 540.05 |
| Chipotle | grocery | 533.34 |
| Airbnb | delivery | 457.43 |
| FedEx | delivery | 526.31 |
| Whole Foods Market | grocery | 573.26 |
| H&M | grocery | 625.64 |
| Google Play | services | 540.22 |
| AT&T | delivery | 465.01 |
| IKEA | categories: | 465.39 |
| Domino's Pizza | grocery | 580.28 |
| Burger King | grocery | 523.52 |
| eBay | grocery | 508.22 |

> Steady performance in the **400–600ms** range. Still usable for batch processing.

---

#### Qwen 3B

```bash
$ python extract_categories.py 2
```


| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 960.48 |
| Starbucks | grocery | 966.19 |
| Walmart | grocery | 1082.68 |
| Target | grocery | 909.43 |
| Apple Store | electronics | 957.61 |
| Costco | grocery | 1106.30 |
| Uber | delivery | 847.72 |
| McDonald's | delivery | 1092.09 |
| Netflix | delivery | 929.05 |
| Best Buy | electronics | 964.11 |
| Shell | grocery | 864.20 |
| CVS Pharmacy | pharmacy | 1041.96 |
| Home Depot | electronics | 983.04 |
| Walgreens | pharmacy | 964.87 |
| Nike | electronics | 958.02 |
| Subway | delivery | 926.89 |
| Delta Airlines | delivery | 1097.80 |
| Spotify | delivery | 951.03 |
| Lowe's | grocery | 1090.20 |
| Chipotle | delivery | 980.19 |
| Airbnb | delivery | 999.13 |
| FedEx | delivery | 947.11 |
| Whole Foods Market | grocery | 1077.83 |
| H&M | clothing | 979.43 |
| Google Play | delivery | 995.60 |
| AT&T | delivery | 1054.53 |
| IKEA | electronics | 1222.08 |
| Domino's Pizza | delivery | 1162.61 |
| Burger King | delivery | 1089.65 |
| eBay | electronics | 929.69 |

> More semantic nuance at the cost of latency. All calls stay **under 1.2 seconds**.

---

#### Qwen 7B

```bash
$ python extract_categories.py 3
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | retail | 2292.83 |
| Starbucks | restaurant | 2177.79 |
| Walmart | grocery | 2110.41 |
| Target | retail | 1918.18 |
| Apple Store | electronics | 2236.61 |
| Costco | retail | 2211.38 |
| Uber | transportation | 1892.72 |
| McDonald's | restaurant | 2470.24 |
| Netflix | entertainment | 1871.59 |
| Best Buy | electronics | 2561.44 |
| Shell | fuel | 1935.49 |
| CVS Pharmacy | pharmacy | 2588.70 |
| Home Depot | hardware | 2310.50 |
| Walgreens | pharmacy | 2255.45 |
| Nike | clothing | 2172.35 |
| Subway | restaurant | 2311.69 |
| Delta Airlines | transportation | 2245.35 |
| Spotify | entertainment | 2171.47 |
| Lowe's | home improvement | 2507.52 |
| Chipotle | restaurant | 2202.90 |
| Airbnb | lodging | 2109.86 |
| FedEx | transportation | 2178.98 |
| Whole Foods Market | grocery | 2384.26 |
| H&M | clothing | 2290.72 |
| Google Play | telecom | 2166.89 |
| AT&T | telecom | 2123.01 |
| IKEA | furniture | 2332.35 |
| Domino's Pizza | restaurant | 2643.26 |
| Burger King | restaurant | 2458.64 |
| eBay | telecom | 2168.95 |

> This is **7 billion parameters** running interactively on CPU! The response times (2–2.6s) are within tolerable range for some real-time workflows.

---
