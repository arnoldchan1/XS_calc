
## Overall goal: Calculate X-ray scattering signal given coordinates.

Equation for X-ray scattering: `S(q, c1, c2) = sum_i sum_j f_i(q, c1, c2), f_j(q, c1, c2) sinc(q * r_ij)`

i, j are atomic indices, f are form factors (FF), sinc is the sin(x)/x function (sinc(0) = 1)

See both CRYSOL and FoXS papers.


### Tasks to do

- Compile atom-specific data: vdW radii and form factors
- Write these functions
  - Function to parse molecular input (we'll use MDAnalysis)
  - SASA calculation (hard)
  - Form factor correction (easy)
  - X-ray scattering signal calculation (that calls SASA calculation and FF correction) (easy)
  - Function to fit calculated signal to experimental signal (medium)





