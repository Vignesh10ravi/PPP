\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-article}[2024-04-18 TUBAF class for articles (based on scrartcl) by Max Weiner]

\DeclareKeys[tubaf-article]{}

\DeclareUnknownKeyHandler[tubaf-article]{%
    \PassOptionsToClass{\CurrentOption}{scrartcl}
}

\ProcessKeyOptions[tubaf-article]

\LoadClass[parskip=half]{scrartcl}

\RequirePackage{tubaf-logo}

\RequirePackage{tubaf-title}
\SetKeys[tubaf-title]{
    black=false,
    trapezoid=false
}

\RequirePackage{tubaf-marks}
\SetKeys[tubaf-marks]{
    footmark=author-title
}

\RequirePackage{tubaf-fonts}
