\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-protocol}[2024-04-18 TUBAF class for protocols (based on tubaf-article) by Marcus Herbig and Max Weiner]

\DeclareKeys[tubaf-protocol]{}

\DeclareUnknownKeyHandler[tubaf-protocol]{%
    \PassOptionsToClass{\CurrentOption}{tubaf-article}
}

\ProcessKeyOptions[tubaf-protocol]

\LoadClass[titlepage=true]{tubaf-article}

\RequirePackage{tabularray}
\RequirePackage{lastpage}
\RequirePackage{xstring}

\providecaptionname{english}{\tubaf@protocol@subject}{Protocol}
\providecaptionname{ngerman}{\tubaf@protocol@subject}{Protokoll}
\subject{\tubaf@protocol@subject}

\providecaptionname{english}{\tubaf@protocol@supervisorname}{\IfSubStr{\tubaf@protocol@supervisor}{\and}{Supervisors}{Supervisor}}
\providecaptionname{ngerman}{\tubaf@protocol@supervisorname}{Betreuer}

\providecaptionname{english}{\tubaf@protocol@coursename}{\IfSubStr{\tubaf@protocol@course}{\and}{Courses}{Course}}
\providecaptionname{ngerman}{\tubaf@protocol@coursename}{\IfSubStr{\tubaf@protocol@course}{\and}{Studiengänge}{Studiengang}}

\providecaptionname{english}{\tubaf@protocol@datename}{Date}
\providecaptionname{ngerman}{\tubaf@protocol@datename}{Datum}

\providecaptionname{english}{\tubaf@protocol@pagecountname}{Page Count}
\providecaptionname{ngerman}{\tubaf@protocol@pagecountname}{Seitenzahl}

\providecaptionname{english}{\tubaf@protocol@authorsname}{\IfSubStr{\tubaf@protocol@author}{\and}{Authors}{Author}}
\providecaptionname{ngerman}{\tubaf@protocol@authorsname}{\IfSubStr{\tubaf@protocol@author}{\and}{Autoren}{Autor}}

\providecaptionname{english}{\tubaf@protocol@matriculationnumbername}{\IfSubStr{\tubaf@protocol@matriculationnumber}{\and}{Matriculation-Numbers}{Matriculation-Number}}
\providecaptionname{ngerman}{\tubaf@protocol@matriculationnumbername}{\IfSubStr{\tubaf@protocol@matriculationnumber}{\and}{Matrikelnummern}{Matrikelnummer}}

\newcommand{\tubaf@protocol@supervisor}{}
\newcommand{\supervisor}[1]{\renewcommand{\tubaf@protocol@supervisor}{#1}}

\newcommand{\tubaf@protocol@course}{}
\newcommand{\course}[1]{\renewcommand{\tubaf@protocol@course}{#1}}

\newcommand{\tubaf@protocol@matriculationnumber}{}
\newcommand{\matriculationnumber}[1]{\renewcommand{\tubaf@protocol@matriculationnumber}{#1}}

\newcommand{\module}[1]{\subtitle{#1}}
\newcommand{\experiment}[1]{\title{#1}}


\titlefoot{
    \renewcommand{\and}{\\}
    \begin{tblr}{ll}
        \tubaf@protocol@authorsname:    & \tubaf@protocol@author     \\
        \tubaf@protocol@matriculationnumbername: & \tubaf@protocol@matriculationnumber \\
        \tubaf@protocol@coursename:     & \tubaf@protocol@course     \\
        \tubaf@protocol@supervisorname: & \tubaf@protocol@supervisor \\
        \tubaf@protocol@datename:       & \tubaf@protocol@date       \\
        \tubaf@protocol@pagecountname:  & \pageref{LastPage}         \\
    \end{tblr}
}

\renewcommand{\tubaf@title@titlepage@prehook}{
    \let\tubaf@protocol@author\@author
    \let\tubaf@protocol@date\@date
    \let\@author\empty
    \let\@date\empty

    \renewcommand{\tubaf@title@titlepage@posthook}{
        \let\@author\tubaf@protocol@author
        \let\@date\tubaf@protocol@date
    }
}