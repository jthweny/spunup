<!-- ✦ you opened the raw source. respect. every [ value ▾ ] on the set-up page can be changed. go poke things. ✦ -->

<div align="center">
<pre>
 ███████╗ ██████╗  ██╗   ██╗ ███╗   ██╗ ██╗   ██╗ ██████╗ 
 ██╔════╝ ██╔══██╗ ██║   ██║ ████╗  ██║ ██║   ██║ ██╔══██╗
 ███████╗ ██████╔╝ ██║   ██║ ██╔██╗ ██║ ██║   ██║ ██████╔╝
 ╚════██║ ██╔═══╝  ██║   ██║ ██║╚██╗██║ ██║   ██║ ██╔═══╝ 
 ███████║ ██║      ╚██████╔╝ ██║ ╚████║ ╚██████╔╝ ██║     
 ╚══════╝ ╚═╝       ╚═════╝  ╚═╝  ╚═══╝  ╚═════╝  ╚═╝     
 ─────────────────────────────────────────────────────────
   ░▒▓█  CHEAP HARDWARE. FULL SPEED. ZERO GUESSWORK.  █▓▒░
 ─────────────────────────────────────────────────────────
</pre>
</div>

<div align="center">

<h1><code>npx spunup-cli</code></h1>

**one command. any terminal. that's the whole install.** ⚡

**pick a model. press Start. that's it.**

*your model, on the cheapest GPUs around or your own computer, squeezed for every last token/s you paid for.*

![where](https://img.shields.io/badge/runs%20on-Local%20%C2%B7%20Vast.ai%20%C2%B7%20RunPod-8a5cf6?style=flat-square)
![recipes](https://img.shields.io/badge/checked%20recipes-100-62e6c9?style=flat-square)
![families](https://img.shields.io/badge/model%20families-24-ff9ad5?style=flat-square)
![flags forgotten](https://img.shields.io/badge/flags%20forgotten-0-000?style=flat-square)
![public urls](https://img.shields.io/badge/public%20URLs-0-000?style=flat-square)
![os](https://img.shields.io/badge/macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-supported-8a5cf6?style=flat-square)

</div>

---

## 👁️ wtf is this

story time. you want to run a model yourself.

```
> rent a GPU
> launch the model
> forgot the tool-call parser
> relaunch
> context too big, out of memory
> relaunch
> caching was off the whole time
> relaunch
> meter's been running for 2 hours
> still not sure it's right
```

never again 🤯

**spunup is the most laid-back way to run a model, ever.** it finds the hardware, picks the runtime, sets every flag, sizes the context, and hands you back a private, OpenAI-compatible endpoint + API key.

**one-shot. every model. first launch.** ⛓️‍💥

no talking to an agent for an hour trying to remember startup flags. no "did i turn on caching?" no relaunching five times while the meter runs. you save your day **and** your money, whether the model costs $3/hr or 30 cents.

---

## ⚡ quick start

got Node.js 22+? one line:

```sh
npx spunup-cli
```

the first run sets up the `spunup` command for good. from then on, from **any** terminal, anywhere on your machine:

```sh
spunup
```

that's the whole tutorial 🤗

no Node? no problem. these bring their own private copy, tucked inside `~/.spunup`, so it never touches anything you already have. no sudo. no admin.

```sh
curl -fsSL https://raw.githubusercontent.com/jthweny/spunup/main/install/install.sh | sh    # macOS / Linux
irm https://raw.githubusercontent.com/jthweny/spunup/main/install/install.ps1 | iex         # Windows (PowerShell)
```

works on **Windows, macOS and Linux.** new version out? `npx spunup-cli@latest` updates it.

---

## 💸 the cheapest hardware, actually used properly

three places to run, and that's all you need:

| | where | why |
|---|---|---|
| 🖥️ | **Local** | your own computer. free. |
| ⚡ | **Vast.ai** | some of the cheapest GPUs you can rent, anywhere |
| 🚀 | **RunPod** | cheap rented GPUs, solid machines |

cheap GPUs are only a bargain if you **use all of them.** most people rent a beast and run it at half speed because one flag was wrong.

spunup doesn't let that happen. **every last bit of performance out of the hardware you can afford. nothing wasted.** 🗜️

---

## 🧠 the recipe database

this is the secret sauce. two layers, both hand-checked:

**1. model recipes** 🎯
the top 100 models, each with the start command that gets the best tokens/s on every kind of hardware. checked against model cards, runtime docs and real runs. every setting has a source link.

**2. family recipes** 🧬
a smart config for each of 24 model families. any model from that family, even one that dropped this morning, gets served the way its family runs fastest.

```
Qwen · DeepSeek · Kimi · GLM · Llama · Mistral · Gemma · gpt-oss · MiniMax · MiMo · Nemotron · Muse
Ornith · Inkling · LFM · Hunyuan · Laguna · Step · Granite · MiniCPM · Ling · Mellum · Nanbeige · Beam
```

model not on the list? **paste any Hugging Face link or name.** it falls back to its family recipe, then to safe general settings. it'll still run 🌱

---

## 🗜️ Smart squeeze: max performance for the price

the set-up page is **one pane.** everything on it starts **✦ smartly tuned** for the most bang for your buck:

- the **most quality per dollar**: the size (quant) and machine that give you the most for every $/hr,
- the **real** quant it picked, shown right there. never a guess,
- the **longest context** that machine can hold.

change anything and the ✦ settings re-tune around you, live. the page tells you exactly what moved.

want that extra bit of quality on this model and don't mind paying for it? bump the size up. your call, always. spunup just makes sure you know what it costs 😏

- 🎛️ anything you can change looks like `[ value ▾ ]`. no hunting.
- 🚫 can't pick something? greyed out **with the reason.**
- 🖥️ want your own machine? pick it. Max context is re-measured on whatever you chose.
- 📏 custom context size? type it in.

---

## 📈 tokens/s you can trust

before you spend a cent, spunup predicts **tokens/s** and **prefill** for the exact combo: model × size (quant) × runtime × hardware.

no vibes. actual math:

```
SPEED     →  bytes of active weights ÷ memory bandwidth
CALIBRATE →  checked against real llama.cpp benchmarks for every GPU
MEASURE   →  your own RAM speed, tested on your own machine
SPLIT     →  GPU + RAM mixed setups, MoE experts placed where they fit
RANGE     →  min to max, because no two rented boxes are the same
```

and when it's running, the final screen shows the speed **measured on the real machine.** receipts 🧾

| run | predicted | measured |
|---|---|---|
| Vast.ai · RTX 3080 · Qwen3.5-2B BF16 · vLLM | 85–113 tokens/s | **86.9** ✅ |
| Windows · RTX PRO 6000 · Qwen3.5-0.8B Q8 | 400–454 tokens/s | **433–444** ✅ |
| cloud VM · 4-core CPU · Qwen3.5-0.8B Q8 | 15.8–17.9 tokens/s | **17.8** ✅ |
| RunPod · RTX 3090 · Magistral-Small Q4_K_M | 33.5–44.6 tokens/s | **54.3** 🤯 beat the estimate |

---

## 🍎 the right quants for YOUR machine

running locally? spunup pulls the format your setup actually wants:

- **Apple Silicon** → MLX or GGUF, sized to the memory your Mac lets the GPU use
- **CPU + RAM only** → GGUF on llama.cpp
- **GPU + RAM mixed** → GGUF, as much on the GPU as fits, MoE experts in RAM
- **Windows without WSL** → llama.cpp, runs natively

no downloading 40 GB only to find out it won't load 🙃

---

## ✅ on by default. always.

the stuff you'd expect to be on? **it's on.** there's no setting, because there's nothing to forget:

- 🔧 **tool calling**, with the right parser for the model. coding agents work straight away.
- 💭 **reasoning/thinking** output parsed properly
- ⚡ **prefix caching**, wherever the runtime supports it
- 🔌 **OpenAI-compatible** endpoint (`/v1/chat/completions`, `/v1/models` …)
- 🔑 **API key** created and required

```
 ✓ Your model is running
   Endpoint   http://localhost:8001/v1
   API key    sk-spunup-••••••••   (press c to copy, k to show)
   Model      qwen3-coder-30b   Q4 · vLLM
   Speed      44 tokens/s   prefill 2,050 tokens/s
   Cost       $0.28/hr · $0.03 so far
```

---

## 🔐 paranoid about privacy. on purpose.

privacy isn't a feature here. it's in every layer of the app:

- runtimes listen on **127.0.0.1 only.** always.
- rented machines are reached through an **SSH tunnel from your computer.** no Cloudflare. no public link. nothing for strangers to find.
- every endpoint **requires its API key.** no key, no tokens.
- your keys live in your **system keychain** where possible.
- **same endpoint every time.** each model keeps its own `localhost` port and key.
- tunnel dropped (sleep, Wi-Fi swap)? it reconnects quietly.
- everything spunup keeps is in `~/.spunup`. nowhere else.

---

## 💰 no surprise bills

- **price per hour** + **cost so far**, for every rented machine, right in the terminal
- **Stop** → the GPU charge stops, storage keeps billing
- **Delete** → everything stops. gone.
- ⏰ **auto-stop when idle**: 30 min, 1 hour, 3 hours, or your own time
- a **Cancel** button the whole way through start-up: stop or delete the machine right there, so a bad launch doesn't keep billing

---

## 🛠️ for the curious

```sh
npm install && npm run build && npm test
```

recipes live in `recipes/`, each one with its source links.

PRs welcome! new model, faster flags, a recipe that's wrong? send it 🫶

---

<div align="center">

*the weights were always yours. now the speed is too.*

```
·•·•✦•·•·  spun up · squeezed · stay private  ·•·•✦•·•·
```

**MIT** · drop a ⭐ if it saved you a relaunch

</div>
