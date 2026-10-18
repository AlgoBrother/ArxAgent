import mayatok as mtk

tokenizer = mtk.get_tokenizer("vb100k")
stament = "This is a test sentence to check the tokenizer functionality."
a = tokenizer.encode(stament)
b = tokenizer.decode(a)
print(a)
print(b)

if b == stament:
    print("Tokenizer is working correctly.")
else:
    print("Tokenizer is not working correctly.")

    # PASS

print("Module:", mtk.__name__)

print("Exports:", [x for x in dir(mtk) if not x.startswith("_")])



for char in ["∇", "≤", "₀", "₁", "‖"]:
    ids = tokenizer.encode(char)
    decoded = tokenizer.decode(ids)

    print(f"\nCharacter: {char!r}")
    print(f"IDs:      {ids}")
    print(f"Decoded:  {decoded!r}")



for char in ["∇", "≤", "₀", "₁", "‖"]:
    ids = tokenizer.encode(char)

    print(f"\nOriginal: {char!r}")
    for token_id in ids:
        print(
            f"  ID {token_id}: "
            f"id_to_token={tokenizer.id_to_token(token_id)!r}, "
            f"get_token_from_id={tokenizer.get_token_from_id(token_id)!r}"
        )

    print("Decoded:", repr(tokenizer.decode(ids)))





    # token

import inspect
import mayatok

tok = mayatok.get_tokenizer("vb100k")

print(tok.add_tokens.__doc__)
print(tok.get_vocab().get("∇"))
print(tok.token_to_id("∇"))
