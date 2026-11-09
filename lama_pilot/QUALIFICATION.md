# LaMa object-removal qualification — development only

The owner-defined foreground annotation for `16pro_187` was saved before these
operation masks. The model is IOPaint's published `big-lama.pt` TorchScript
checkpoint, SHA-256
`344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`;
its official MD5 is `e3aa4aaa15225a33ec84f9f4bc47e500`.
All source, mask, output, and checkpoint hashes are in the trial JSON files.
The output is **not** part of a held-out evaluation set.

| Trial | Realised area and location | Visual assessment before detector scores | Decision |
|---|---|---|---|
| v1 scooter mirror | 0.450%, foreground | Mirror remained visible after inpainting. | Reject as a removal edit. |
| v1 two driveway cones | 0.420%, background | Cones mostly removed, but yellow cone tips remained outside the mask. | Reject. |
| v2 scooter mirror | 1.635%, foreground | Mirror remained and a blurred bar appeared near its attachment. | Reject. |
| v2 two driveway cones | 0.671%, background | Cones are absent and paving texture is continuous at normal viewing size; a small shadow or seam may remain in an enlarged crop. Source and mask geometry match, and the localized output leaves all outside-mask pixels byte-identical. | Development candidate only; owner visual confirmation pending. |

The v2 masks were revised **after looking only at v1 editing quality**, before
any detector scores. They were not selected using detector output. The
foreground and background v2 areas are not closely matched, so this pair
cannot support a size-controlled location comparison. The candidate is an
object-removal edit, which should be analysed separately from text-guided
object insertion if both are later used. Keep all rejected trials in the
qualification log and out of the accepted research sample manifest.

The LaMa result suggests mask-conditioned removal is more promising locally
than the five unsuccessful SD v1.5 text-guided insertion trials. A single
candidate does not establish a reliable editor across other roots, area
bands, or foreground locations.
