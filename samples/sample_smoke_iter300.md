# Sample — this is smoke checkpoint (iter 300, val loss 3.85)

This is the **under-trained** CPU smoke model from Phase 3 (only 300 iters,
val loss 3.85, perplexity ≈ 47). It already produces real TinyStories vocabulary
and structure, but rambles and drops grammar over longer spans — exactly what a
loss of ~3.85 looks like. The full 5000-iter Colab run (target val loss ~1.7,
perplexity ≈ 5.5) will produce coherent stories; this file is the "before".

Command:
```
python -m src.generate --prompt "Once upon a time" \
    --max-new-tokens 150 --num-samples 2 --temperature 0.8 --top-k 200 --seed 1234
```

Loaded `checkpoints/best.pt` — iter 300, best val loss 3.8464, device=cpu.

---

**Sample 1** (temperature 0.8, top-k 200)

> Once upon a time, there was a puppy. she wanted to mix it. One day, he liked to
> help. He ran to play with the way to pick it up and a little boy was so he got
> to watch and said, "Mom sounds. "Why can't want, "Let's go for a noise, "I will
> share. I want?" The boy's be careful." "Yes, Tom. I'm we're welcome, "I'm sorry,
> Anna said, I don't be careful and Dad said, this very sad. Can you have to the
> swings and said. They could see what the rock.

**Sample 2** (temperature 0.8, top-k 200)

> Once upon a time, there was a little girl named Lily. She loved to the bird. So
> the ball and put it. She felt sad and played over and play with her mom said,
> "That's special and she was a bag. Tom asked her mom in the park and ate it,
> "What love you are very sad. I am so scared for the doll, "No, "I'm sorry to the
> sky was so tired, "I must give you. I lost me." When I will be like a big tree.
> I'm so she said, "I play with her room and said, "I'm going to help her mom."

_What's already learned:_ the "Once upon a time, there was a … named …" opening,
character names (Lily, Tom, Anna), a children's-story register, dialogue with
quotation marks, simple emotions ("felt sad"). _What's missing at val 3.85:_
consistent grammar, plot coherence, and staying on a single thread — these come
as the loss falls toward ~1.7.
