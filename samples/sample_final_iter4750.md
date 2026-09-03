# Samples — final model (iter 4750, val loss 1.88)

The fully-trained model from the 5000-iter Colab (T4) run — val loss **1.877**,
perplexity ≈ **6.5**. Compare with [`sample_smoke_iter300.md`](sample_smoke_iter300.md)
(the under-trained val-3.85 "before"): coherent plots, consistent characters, and
proper grammar now emerge.

Command:
```
python -m src.generate --prompt "Once upon a time" \
    --num-samples 3 --max-new-tokens 200 --temperature 0.8 --top-k 200 --seed 42
```

Loaded `checkpoints/best.pt` — iter 4750, best val loss 1.8770.

---

**Sample 1** (temperature 0.8, top-k 200)

> Once upon a time, there was a little box. Inside the box was a box with a strange
> inside. The box was full of old things. The little girl was so happy and she
> couldn't stop jumping inside. But then a little girl came over and said, "Hello!
> What are you doing?" The little girl replied, "I am searching for the box. I need
> my help." ... Inside was a beautiful box full of colorful colors. She thanked the
> box "Thank you so much for helping me find me in the box".

**Sample 2** (temperature 0.8, top-k 200)

> Once upon a time, there was a little girl named Lily. She loved to play with her
> dolls and stuffed animals every day. One day, Lily's mom took her to a big store.
> Lily saw an incredible toy she really wanted. She asked if she could borrow it.
> Her mom said yes, so Lily was so happy. Lily took the toy and it was so pretty and
> she had lots of fun playing with it. But then, Lily got sad because she couldn't
> play with it. Her mom tried to comfort her... Her mom smiled and said, "I love you
> too, Lily." And they both went back to playing with the toy car.

**Sample 3** (temperature 0.8, top-k 200)

> Once upon a time, there was a little girl named Lily. She loved to play with her
> jewelry and pretend she was a princess. One day, she went to the park to play with
> her jewelry. But when she saw her toy box, a big dog tried to take it from her. The
> dog tried to take her jewelry away from her. Lily was sad and started to cry. Her
> mommy told her not to worry and that the dog was not a real friend... Later that
> day, Lily's friend came over to play. Lily was so happy to have her shiny jewelry
> box back. She promised to always be careful and respectful with it. From that day
> on, Lily and her friends played with her jewelry box, and the dog never grabbed it
> again.

_Observations:_ character names carry through, plots have a beginning/conflict/
resolution, dialogue is punctuated correctly, and stories end with the little moral
TinyStories is known for. Sample 1 shows the model's remaining weakness — it can get
stuck repeating a noun ("box") — which larger models / more training would smooth out.
