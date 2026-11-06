# Pitch transcript: "A tiny quantum circuit that tunes a quantum-secure link"

For `pitch.html`, 13 slides plus an appendix, about 10–11 minutes. Plain language; technical detail is in `slides.html`.

[Brackets] are stage notes. **PENDING** marks lines to finish once the benchmark run is done. Each pending line has
both versions, so you can read the true one.

---

## Slide 1 · Title (40 s)

Quantum key distribution, QKD, lets two places share a secret key over an optical fiber. Its security rests on the
laws of physics, not on how hard some maths problem is.

Why does that matter? QKD is the only way to share keys whose security rests on physics rather than on hard maths
problems, so it stays safe even against future computers. But it has a quiet failure mode: with the wrong settings, a
link can produce zero key without anything visibly breaking.

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

[point at the three cards]
- **61,875 verified examples.** Each example is one link, a fiber length and a session size, with its five best
  settings. That's 15 session sizes times about 4,000 fiber lengths.
- **Physics bug fixed.** The earlier key-rate model had a bug. Fixed, it now matches the published reference within
  about 8 %.
- **Built in 4.6 minutes** on a laptop.

## Slide 4 · What has to be learned (50 s)

Here are those best settings: one panel per session size, five coloured lines for the five knobs.

[sweep left to right] At short and medium distance the settings are almost flat. You could nearly hard-code them.

[point at the right edge] Near the maximum distance everything moves fast. The basis probability, the red line, drops
to zero. The signal probability, the green line, shoots up.

[point at a triangle] The small triangles mark places where the best setting switches between two options that are
almost equally good. The key rate itself doesn't jump there; only the recipe changes.

So a good tuner has to be easy in the middle, accurate at the edge, and careful at the switches.

## Slide 5 · The challenge (60 s)

So why not just store a big table of answers and look them up?

One: the edge is unforgiving. Near the maximum distance the key falls about a million-fold in the last 15 kilometres,
and small mistakes in the settings can cost all of it. There, even our earlier table was exactly right only about one
time in four.

Two: the recipe sometimes jumps. At a dozen or so distances, the best settings switch suddenly to a different recipe
that's just as good. Averaging the two recipes gives a bad one.

Three: real links have more knobs than our two inputs. Detector noise, alignment, fiber quality. Every extra knob
multiplies the size of a table, while a learned model stays small.

To be fair to the table: with today's two inputs a big table still works well, and we compare against it honestly.
The learned tuner pays off as inputs are added.

[point at the blue box] So we need a learned function, not a table. And the question for the quantum circuit is
whether it is a *good* function for this job, not whether it is *faster*.

## Slide 6 · Our quantum idea (70 s)

Here is our circuit. It's deliberately tiny: five qubits.

Five qubits always carry the five settings: we measure each of them and read off one setting, and one run of the
circuit gives all five at once. We also tested giving the circuit extra "helper" qubits with no answer of their own,
just more room to think before deciding. The best circuit found uses 8 qubits: 5 answers plus 3 helpers. [point at
the helper wires at the bottom] The read-out is built so that it can only produce allowed settings. The circuit can
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

## Slide 7 · Why 8 qubits × 6 layers (70 s)

[point at the left] What is a "layer"? One block of: feed in the inputs, apply some learned turns, link the qubits.
The six layers are applied one after another to the same eight qubits, like six blocks in a chain, and then we measure
once. It's not six runs that get averaged.

Why feed the inputs in six times? Each time the inputs go in, the circuit's output can contain one more "wave", so it
can bend more sharply. That's exactly what it needs near the cliff at the maximum distance. In our tests, six layers
lost three to nine times less key than three layers with the same number of qubits.

What does it learn? To copy the teacher's five settings as closely as possible. The key rate is how we grade it, and
how we picked the winning shape.

How many numbers does it learn? Eight qubits times six layers gives 48 slots. Each slot learns five numbers: a scale
for the distance, a scale for the session size, and three rotation angles. 48 times 5 is 240.

[point at the table] "Key lost" means: put the predicted settings into the exact physics, compare with the key from
the best settings, and take the shortfall. The median is over about 4,700 validation links held back from training.
We tried six shapes and kept the one that lost the least: 8 qubits by 6 layers, at 0.54 %.

Why three helpers and not one, two or four? Honestly, we tested only zero, one and three, because every extra qubit
doubles the cost of simulating the circuit; the 8-qubit version already takes over 20 minutes to train. One helper
didn't help; three did. But the 8 × 6 circuit also has the most numbers to learn, so we can't yet say whether the
helpers or the size made the difference. Two and four helpers are the next experiment.

## Slide 8 · Teacher, student, exam (50 s)

Think of it as a classroom.

The **teacher** is our exact solver: slow-ish, but always right. It writes a **textbook** of 61,875 worked examples:
"for this link, these are the best five settings."

The **student** studies the textbook. The student is either our quantum circuit or a classical neural network. After
studying, it can answer instantly: give it a link, and it gives five settings.

Then the **exam**: links the student has never seen. The student answers, and the answer is always a valid set of
settings.

And the **grading**, in orange: we don't ask the student how well it did. We plug its answer into the exact physics
and compare the key it produces with the teacher's key. That ratio is the grade.

## Slide 9 · How we test it fairly (45 s)

The test has three steps.

One: train. The quantum circuit, a classical neural network with the same number of adjustable parameters, and a
lookup table, all on the same examples with the same training budget.

Two: predict. They predict settings for distances and session sizes they have never seen.

Three: check, with the physics check from the previous slide.

Training time on a laptop: the simulated quantum circuit takes from under a minute to about 23 minutes depending on
its size. The classical network takes seconds. [PENDING: say the measured NN time.]

We wrote these rules down before training started. So if the quantum circuit ties or loses, that's a valid result,
and we'll report it.

## Slide 10 · Scoreboard (70 s)

[Preliminary numbers from held-back validation links and one training run. Replace with the final test numbers when
the run finishes; the story is unlikely to flip, but say "preliminary".]

Top row: our exact solver. 100 % of the best key, in 1.6 to 30 milliseconds. That's the bar.

The dense lookup table is essentially perfect at typical distances. Near the maximum distance it leaves fewer than
1 % of links without key. With only two inputs, a dense table is very hard to beat; we said so up front.

The classical neural network delivers 99.96 % of the key at typical distances, but near the edge it leaves about 9 %
of links with no key.

Our quantum circuit delivers 99.7 % at typical distances and 96.4 % near the edge, and leaves about 12 % of links near
the edge with no key. On this problem, the classical models win.

Two things are encouraging, though. First, the connections matter: the same circuit without them did clearly worse.
Second, the fix-up step: take the circuit's answer and run one exact correction step, under a millisecond. That brings
it to 99.998 % at typical distances and 99.7 % near the edge.

So the honest summary: as a stand-alone tuner the quantum circuit doesn't beat the classical options yet. As a quick
first guess plus a fix-up, it gets very close to the best key.

## Slide 11 · Showcase (40 s)

[PENDING: show the figure once drawn]

This is a session size none of the models saw during training. The black line is the best setting. Blue is the
quantum circuit, orange the neural network, green the table. On the right: how much of the best key each one delivers
along the fiber.

[point at the edge] Watch the last few kilometres. That's where the models separate.

## Slide 12 · Why it matters (50 s)

Why should anyone care?

Real links drift. Temperature, re-routing and changing traffic alter the fiber and the session size. Fixed settings
waste key every hour of every day.

It matters most near the edge. Close to the maximum distance, the wrong settings can drop the key to zero where the
right ones still produce it. That's the difference between a link that works and one that doesn't.

A fast, learned tuner can retune every link in milliseconds, and it scales to more inputs where tables can't. Looking
further ahead, quantum networks will have small quantum processors in their nodes, and a tuner like this is a natural
first job for them.

## Slide 13 · Next (30 s)

Next steps.

We finish the benchmark that is running now and fill in the scoreboard.

We run the trained circuit on real quantum hardware from IQM, once the budget is approved.

We add more knobs, noise, alignment and fiber loss, where tables stop scaling.

And we build the best of both worlds: a learned first guess plus one exact fix-up step.

Thank you. Happy to take questions.

---

## Slide 14 · Appendix (not presented)

Notes and sources for every [n] mark on the slides. Show it only if someone asks where a number comes from.

---

### Anticipated questions

- **"Are you claiming a quantum advantage?"** No. The circuit runs on a classical simulator, and we compare it with a
  classical network of the same size. We are testing whether it is a good function for this job, not whether it is faster.
- **"Why not just use a bigger table?"** With two inputs a dense table does well. But near the maximum distance even our
  earlier table was exactly right at only about 25 % of points, and every extra input multiplies a table's size.
- **"Has it run on real quantum hardware?"** Not yet. Hardware runs are paused and nothing has been submitted. It's the
  next step, once the budget and the run plan are approved.
- **"What if the quantum circuit loses?"** Then we report it. The rules were fixed before training, so a loss is a
  valid result. And a rough guess is still useful: in our pilot, an error of 1 % of each setting's range cost about
  0.4 % of the key, and two exact fix-up steps (1.4 ms) brought it back to within 2 parts per billion (median).
- **"Why not always use the exact solver?"** You can. It takes 1.6 ms on a running link and 30 ms on a new one. A
  learned tuner matters as an instant first guess, and as inputs are added.
