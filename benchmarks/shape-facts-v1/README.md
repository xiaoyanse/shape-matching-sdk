# Shape facts fixture v1

This directory is the fixed input package for the shape-facts v1
acceptance fixture. Point `TemplateAsset::load` at the absolute path to
`template-and-frame-spec-v1.json`; keep the adjacent `source/` files in place.

The package is a reproducibility fixture, not a production template bank.
`TemplateAsset::load` validates the spec SHA-256 and the two source-file
SHA-256 values before accepting the spec. The native matcher trains its banks
from these source files at runtime. The experiment-generated
`generated-shape-bank-v1.bin` is deliberately excluded because it is not a
production template asset.

## Provenance and license

The JSON and four PNG fixtures originated from the 2026-09 migration
validation asset preparation, preserved
from the project POC `preserved-shape-sop-chain` at upstream HEAD
`890d7ca24b7e20c205c45db7fd416c3416c27460`. Git normalized only the
JSON line endings to LF; its committed byte hash is recorded below. The PNGs are deterministic
program-generated samples; they contain no field imagery. The C++ source and
header are the POC's `source/vision_facts.cpp` and `.h`, retained because the
loader pins their hashes and the native implementation uses them to train.
The POC package does not declare a separate license for these fixture files
or source files; no independent license grant is asserted here. The native
third-party matcher retains its own notices in the parent directory.

The original `frames/field-one-original.jpg` is excluded: the available
provenance does not establish permission to redistribute its field image or
confirm that it contains no privacy-sensitive information. Its preserved
manifest entry is recorded below for identification only.

## Files and SHA-256

| Path | Bytes | SHA-256 |
|---|---:|---|
| `template-and-frame-spec-v1.json` | 1556 | `BDC704555EEDA5BA55C58896C4BA6269210691D85AF4263D2948FCE51378AE9C` |
| `source/vision_facts.cpp` | 6345 | `FD83BAD74F900059F8A505BA9B080F2238B33CA7C1E79CEB9502DA394AD6F382` |
| `source/vision_facts.h` | 2265 | `418FB66AF43A31DBF31833A81B66A9B21E49AA62F56DF74033A6944446DF234F` |
| `frames/synthetic-f0001-positive.png` | 97932 | `9C87D0C9B81D7724C93FBE71FA2B4BCFDA160E8E2B054C66C66AB450E9E232F4` |
| `frames/synthetic-f0002-anchor-absent-target-positive.png` | 95527 | `0FB62835D08F2F77367E166912D60972449855F5CFF6552A0DFF7E841E11017B` |
| `frames/synthetic-f0003-target-negative.png` | 98317 | `250CDDF7BF145D6AA326E6F2B9F08C99FBBF4C8247C411343357151661253A25` |
| `frames/synthetic-f0004-positive-new-pose.png` | 98314 | `034483ECD84B5A142764E4986F62C8BBDFC4CF8F90F6B0662678BB27E0F6A5A9` |

Excluded field image SHA-256 (from the preserved source manifest):
`5686E4C70FDF71F02C3D6B9D7AE04D197400C4CBB92F2CF7E3204DDB51332A56`
(`frames/field-one-original.jpg`, 649364 bytes). It remains only in the E: drive
evidence package.

The four frame names and behavioral labels follow the preserved POC README.
They describe that POC's outputs on fixed inputs and are not independent
annotations or field-accuracy evidence. The spec fixes the generated template
contract (540×540 canvas, 260×260 crop at (140,140), rotations −25° through
25° in 5° increments, threshold 70). This is a contract for reproducible
matcher wiring, not evidence that these synthetic shapes are appropriate for
production.
