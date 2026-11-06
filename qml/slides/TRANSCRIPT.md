# Pitch transcript: "A tiny quantum circuit that tunes a quantum-secure link"

For `pitch.html`, 12 slides, about 9–10 minutes. Plain language; technical detail is in `slides.html`.

[Brackets] are stage notes. **PENDING** marks lines to finish once the benchmark run is done. Each pending line has
both versions, so you can read the true one.

---

## Slide 1 · Title (40 s)

Quantum key distribution, QKD, lets two places share a secret key over an optical fiber. Its security rests on the
laws of physics, not on how hard some maths problem is.

There is a practical catch. The sender's laser has five settings, and the right values depend on the link. How long
is the fiber? How much data are we collecting in one session? Get the settings wrong and you throw away secret key.

Our question today: can a tiny quantum circuit learn to set those five knobs? And if it can, is it any better than a
classical AI doing the same job?

Two numbers we already have. Our classical solver finds the best settings in every case we checked, and it retunes a
link in under two milliseconds. The third number, how much key the quantum circuit delivers, is what we're measuring
right now.

## Slide 2 · The problem (50 s)

This chart shows how much secret key you get per laser pulse as the fiber gets longer, with the best possible
settings. Each line is a different session size, meaning how much data we collect before making a key.

The key drops about ten times every fifty kilometres. That's just fiber loss.

[point at the triangles] Each line also hits a hard wall, a maximum distance. For small sessions it's 130 km; for
very large ones, 184 km. Past that wall there's no key at all.

Getting close to the wall needs the right five settings for that exact distance. So this is a tuning problem, and it
has to be solved again whenever the link changes.

## Slide 3 · What we built: the classical solver (60 s)

Before teaching any AI, we needed the right answers to teach it.

The earlier version of this project used a general-purpose search. When we checked it carefully, it found the truly
best settings in only about 4 % of cases, and it took 6 to 18 seconds per link.

We built an exact solver. Two situations:
- A brand-new link, where we know nothing in advance. It finds the best settings in about 30 milliseconds.
- A link that's already running and changes a little, say the temperature shifts. It starts from the current
  settings and is done in 1.6 milliseconds.

Both find the best settings every time we checked, against an independently built reference.

With this solver we built our dataset: 61,875 examples. Each example is one link, a fiber length and a session size,
together with its five best settings. That's 15 session sizes times about 4,000 fiber lengths, and it took
4.6 minutes on a laptop.

We also found and fixed a bug in the earlier physics model. It now matches the published reference within 8 %.

## Slide 4 · All five settings across distance (50 s)

Here are those best settings: one panel per session size, five coloured lines for the five knobs.

[sweep left to right] At short and medium distance the settings are almost flat. You could nearly hard-code them.

[point at the right edge] Near the maximum distance everything moves fast. The basis probability, the red line, drops
to zero. The signal probability, the green line, shoots up.

[point at a triangle] The small triangles mark places where the best setting switches between two options that are
almost equally good. The key rate itself does not jump there; only the recipe changes.

So a good tuner has to be easy in the middle, accurate at the edge, and careful at the switches.

## Slide 5 · The challenge (50 s)

So why not just store a big table of answers and look them up?

One: the edge is unforgiving. Near the maximum distance, small mistakes in the settings can cost most of the key.
There, a simple table is exactly right only about one time in four.

Two: the recipe sometimes jumps. At a dozen or so distances, the best settings suddenly switch to a different recipe
that's just as good. If a model averages the two recipes, it gets a bad one.

Three: real links have more knobs than our two inputs. Detector noise, alignment, fiber quality. Every extra knob
multiplies the size of a table, while a learned model stays small.

To be fair to the table: with today's two inputs, a big table still works well, and we compare against it honestly.
The learned tuner pays off as inputs are added.

## Slide 6 · Our quantum idea (70 s)

Here is our circuit. It's deliberately tiny: five qubits.

Why five? Because there are five settings. We measure each qubit and read off one setting, and one run of the
circuit gives all five at once. The read-out is built so that it can only produce allowed settings. The circuit can
be inaccurate, but never invalid.

Why the connecting wires between the qubits? Because the settings are connected.
- The rules tie them: the signal pulse must be stronger than the decoy, and the probabilities must share 100 %.
- The data shows it too. Across distance, the signal probability and the basis choice move in lock-step, a
  correlation of minus 0.97. The decoy strength and the decoy probability move together at plus 0.97.

The wires let the qubits share that information.

But we don't just assume the wires help. We train the identical circuit without any wires as well. If it does just
as well, the wires weren't needed, and we'll say so.

What we do not claim: no quantum advantage and no speed-up on today's hardware. The circuit runs on a simulator. The
question is how accurately it learns, for its size.

## Slide 7 · How it works (50 s)

Here's the whole pipeline.

Top row: the exact solver generates the 61,875 examples. We train a tuner on them, either the quantum circuit or a
classical neural network, and get a trained tuner that maps a link to five settings.

Bottom row: a new link comes in, with its fiber length and session size. We ask the tuner, which is one quick
prediction, and get five settings that are always valid.

Then the important part, in orange: the physics check. We plug the predicted settings into the exact physics and
compare the key they produce with the key from the best settings. That ratio is our score. We never use a model's
own estimate of how good it is.

## Slide 8 · How we test it fairly (45 s)

The test has three steps.

One: train. The quantum circuit, a classical neural network with the same number of adjustable parameters, and a
lookup table, all on the same examples with the same training budget.

Two: predict. They predict settings for distances and session sizes they have never seen.

Three: check, with the physics check from the previous slide.

Training time on a laptop: the simulated quantum circuit takes from under a minute to about 23 minutes depending on
its size. The classical network takes seconds. [PENDING: say the measured NN time.]

We wrote these rules down before training started. So if the quantum circuit ties or loses, that's a valid result,
and we'll report it.

## Slide 9 · Scoreboard (60 s)

[PENDING: read the row values once filled]

Top row: our exact solver. 100 % of the best key, in 1.6 to 30 milliseconds. That's the bar.

The question for each learned model: how much of that key does it deliver, at typical distances and near the edge,
and how fast is each answer?

**PENDING. If the quantum circuit is close to the neural network:** "At the same size, the quantum circuit delivers
X % of the best key, similar to a classical neural network with Y %. So it learns this problem about as well. We
don't see an advantage, and we don't see a disadvantage."

**PENDING. If it is clearly behind:** "The quantum circuit delivers X %, behind the classical network's Y %. On this
small problem, the classical model wins."

**PENDING. If it is ahead:** "The quantum circuit delivers X % against Y % for a network of the same size. That's an
encouraging sign for its accuracy per parameter. It's a simulation result, not a speed claim."

The last row is the practical trick. Take the circuit's guess and add one exact fix-up step from the solver. In our
pilot, a guess that is 1 % off loses about 0.4 % of the key. Two fix-up steps, about one and a half milliseconds,
bring it back almost exactly to 100 %.

## Slide 10 · Showcase (40 s)

[PENDING: show the figure once drawn]

This is a session size none of the models saw during training. The black line is the best setting. Blue is the
quantum circuit, orange the neural network, green the table. On the right: how much of the best key each one delivers
along the fiber.

[point at the edge] Watch the last few kilometres. That's where the models separate.

## Slide 11 · Why it matters (50 s)

Why should anyone care?

Real links drift. Temperature, re-routing and changing traffic alter the fiber and the session size. Fixed settings
waste key every hour of every day.

It matters most near the edge. Close to the maximum distance, the wrong settings can drop the key to zero where the
right ones still produce it. That's the difference between a link that works and one that doesn't.

A fast, learned tuner can retune every link in milliseconds, and it scales to more inputs where tables can't. Looking
further ahead, quantum networks will have small quantum processors in their nodes, and a tuner like this is a natural
first job for them.

## Slide 12 · Next (30 s)

Next steps.

We finish the benchmark that is running now and fill in the scoreboard.

We run the trained circuit on real quantum hardware from IQM, once the budget is approved.

We add more knobs, noise, alignment and fiber loss, where tables stop scaling.

And we build the best of both worlds: a learned first guess plus one exact fix-up step.

Thank you. Happy to take questions.

---

### Likely questions (short answers)

- **"Is this a quantum advantage?"** No, and we don't claim one. The circuit runs on a classical simulator. We are
  testing whether it learns this problem as well as a classical model of the same size.
- **"Why not just use the solver?"** You can. It takes 1.6–30 ms. A learned tuner is about cheap instant answers,
  scaling to many inputs, and running where a full solver can't, such as small controllers.
- **"How many shots would real hardware need?"** About a thousand per prediction for 1 %-level accuracy, and about
  100,000 for 0.1 %. That's why the fix-up step matters.
- **"Why did the old method get only 4 %?"** It was a general-purpose search with loose settings, and the physics model
  also had a bug. Both are fixed and documented.
- **"Why are the qubits connected?"** The settings depend on each other: physics rules, plus correlations of about
  0.97 in the data. We also train the same circuit without connections to test whether it matters.
- **"Why does training the quantum circuit take minutes when the NN takes seconds?"** On a laptop we simulate the
  quantum state exactly. Each extra qubit doubles that work. On real hardware, the cost moves to the number of
  measurements instead.
