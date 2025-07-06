# 2014_ConciseSecurityBoundsforPracticalDecoy_State_QKD

PHYSICAL REVIEW A 89, 022307 (2014)
Concise security bounds for practical decoy-state quantum key distribution
Charles Ci Wen Lim,1,*Marcos Curty,2Nino Walenta,1Feihu Xu,3and Hugo Zbinden1
1Group of Applied Physics, University of Geneva, Switzerland
2EI Telecomunicaci ´on, Department of Signal Theory and Communications, University of Vigo, Spain
3Center for Quantum Information and Quantum Control, Department of Physics and Department of Electrical & Computer Engineering,
University of Toronto, Canada
(Received 30 November 2013; revised manuscript received 14 January 2014; published 10 February 2014)
Due to its ability to tolerate high channel loss, decoy-state quantum key distribution (QKD) has been one of
the main focuses within the QKD community. Notably, several experimental groups have demonstrated that it issecure and feasible under real-world conditions. Crucially, however, the security and feasibility claims made bymost of these experiments were obtained under the assumption that the eavesdropper is restricted to particulartypes of attacks or that the ﬁnite-key effects are neglected. Unfortunately, such assumptions are not possibleto guarantee in practice. In this work, we provide concise and tight ﬁnite-key security bounds for practicaldecoy-state QKD that are valid against general attacks.
DOI: 10.1103/PhysRevA.89.022307 PACS number(s): 03 .67.Dd,03.67.Hk
I. INTRODUCTION
In 1984, Bennett and Brassard proposed a quantum key
distribution (QKD) scheme in which a cryptographic key canbe securely distributed between two remote parties, Alice andBob, in an untrusted environment [ 1]. Since then, this proposal
(traditionally referred to as the BB84 protocol) has receivedconsiderable attention, and signiﬁcant progress has been madein both theory and practice [ 2].
In actuality, implementations of the BB84 protocol differ in
some important aspects from the original theoretical proposal.This is particularly the case in the choice of the quantuminformation carrier, where a weak pulsed laser source isused in place of an ideal single-photon source (which is notyet available). However, pulsed laser sources have a criticaldrawback in that a non-negligible fraction of the emitted laserpulses contain more than one photon—which an adversary,Eve, can exploit via the so-called photon-number-splitting(PNS) attack [ 3]. In fact, this attack has been shown to be
extremely powerful, especially when the loss in the quantumchannel connecting Alice and Bob is high.
To tackle the PNS attack in the presence of high channel
loss, most BB84 implementations (e.g., see Refs. [ 4–14]) adopt
the decoy-state method [ 15–17]. The basic idea is conceptually
very simple, and more importantly, it requires minimal modiﬁ-cation to existing BB84 implementations. Speciﬁcally, insteadof preparing phase-randomized laser pulses of the same meanphoton-number, Alice varies randomly and independently themean photon number of each laser pulse she sends to Bob.Crucially, by using the fact that the variation of the meanphoton number is inaccessible to Eve, it is possible to detectthe presence of photon-number-dependent loss in the quantumchannel, i.e., by analyzing the data shared between Alice andBob. As a result, photon-number-dependent type of attacks arecircumvented, and the secret key rates and the tolerance to thechannel loss are signiﬁcantly improved.
The security of decoy-state QKD has been obtained in the
asymptotic regime [ 16,17], i.e., in the limit of inﬁnitely long
keys. In the case of ﬁnite-length keys, several attempts have
*Corresponding author: charles.lim.geneva@gmail.combeen made (e.g., see Refs. [ 18–23]), but most (if not all) of
these results assume that Eve is restricted to particular types
of attacks. Very recently, ﬁnite-key security bounds againstgeneral attacks have been derived by Hayashi and Nakayama[24], although the security analysis is rather involved.
In this work, we provide concise and tight ﬁnite-key
security bounds for a practical decoy-state QKD protocolthat are directly applicable to most current decoy-state QKDimplementations. The security analysis is based on a com-bination of a recent security proof technique [ 25,26] and a
ﬁnite-size analysis for the decoy-state method, which allowsus to greatly simplify the security analysis. As a result,
we are able to derive tight ﬁnite-key security bounds that
are valid against general attacks. Moreover, these boundscan be straightforwardly computed with just ﬁve conciseformulas [see Eqs. ( 1)–(5)], which experimentalists can readily
use for their implementations. In addition, we evaluate theperformance of our security bounds by applying them toa realistic ﬁber-based system model. The evaluation shows
that our security bounds are relatively tight, in the sense that
for realistic postprocessing block sizes, the achievable secretkey rates are comparable to those obtained in the asymptoticregime. In fact, for small postprocessing block sizes (of theorder of 10
4bits), we observe that secret keys can be securely
distributed over a ﬁber length of up to 135 km.
II. PROTOCOL DESCRIPTION
We consider an asymmetric coding BB84 protocol [ 27], i.e.,
the bases XandZare chosen with probabilities that are biased.
Speciﬁcally, the bases XandZare selected with probabilities
qxand 1−qx, respectively, and the secret key is extracted from
the events whereby Alice and Bob both choose the Xbasis. In
addition, the protocol is based on the transmission of phase-randomized laser pulses, and uses two-decoy settings. The in-tensity of each laser pulse is randomly set to one of the three in-tensities μ
1,μ 2, andμ3, and the intensities satisfy μ1>μ 2+
μ3andμ2>μ 3/greaterorequalslant0. Note, however, that our analysis can also
be straightforwardly generalized to any number of intensitylevels. Next, we provide a detailed description of the protocol.
1. Preparation. Alice chooses a bit value uniformly at ran-
dom and records the value in y
i. Then, she selects a basis choice
1050-2947/2014/89(2)/022307(7) 022307-1 ©2014 American Physical Society

LIM, CURTY , W ALENTA, XU, AND ZBINDEN PHYSICAL REVIEW A 89, 022307 (2014)
ai∈{X,Z}with probabilities qxand 1−qx, respectively, and
an intensity choice ki∈K:={μ1,μ2,μ3}with probabilities
pμ1,pμ2, andpμ3=1−pμ1−pμ2, respectively. Finally, she
prepares a (weak) laser pulse based on the chosen values andsends it to Bob via the quantum channel.
2. Measurement. Bob chooses a basis b
i∈{X,Z}with
probabilities qxand 1 −qx, respectively. Then, he performs
a measurement in basis biand records the outcome in y/prime
i.
In practice, the measurement device is usually implementedwith two single-photon detectors. In this case, there are fourpossible outcomes {0,1,∅,⊥}where 0 and 1 are the bit values,
and∅and⊥are the no detection and double detection events,
respectively. For the ﬁrst three outcomes, Bob assigns what heobserves to y
/prime
i, and for the last outcome ⊥he assigns a random
bit value to y/prime
i.
3. Basis reconciliation. Alice and Bob announce their basis
and intensity choices over an authenticated public channel andidentify the following sets: X
k:={i:ai=bi=X∧ki=k∧
y/prime
i/negationslash=∅ } andZk:={i:ai=bi=Z∧ki=k∧y/prime
i/negationslash=∅ } for all
k∈K. Then, they check for |Xk|/greaterorequalslantnX,kand|Zk|/greaterorequalslantnZ,kfor
all values of k. They repeat steps 1–3 until these conditions
are satisﬁed. We denote as Nthe number of laser pulses sent
by Alice until the conditions are fulﬁlled.
4. Generation of raw key and error estimation. First, a
raw key pair ( XA,XB) is generated by choosing a random
sample of size nX=/summationtext
k∈KnX,kofX=∪k∈KXk, where nXis
the postprocessing block size. Note that we use all intensitylevels for the key generation, while existing decoy-state QKDprotocols typically use only one intensity level. Second, theyannounce the sets Z
kand compute the corresponding number
of bit errors, mZ,k. Third, they calculate the number of vacuum
events sX,0[Eq. ( 2)] and the number of single-photon events
sX,1[Eq. ( 3)] in ( XA,XB). Also, they calculate the number of
phase errors cX,1[Eq. ( 5)] in the single-photon events. Finally,
they check that the phase error rate φXis less than φtolwhere
φtolis a predetermined phase error rate, φX:=cX,1/sX,1<φ tol.
If this condition is not met, they abort the protocol, otherwisethey proceed to step 5.
5. Postprocessing. First, Alice and Bob perform an error-
correction step that reveals at most λ
ECbits of information. In
this step, we assume that they try to correct for an error ratethat is predetermined. Next, to ensure that they share a pairof identical keys, they perform an error-veriﬁcation step usingtwo-universal hash functions that publishes ⌈log
21/εhash⌉bits
of information [ 28]. Here, εhashis the probability that a pair
of nonidentical keys passes the error-veriﬁcation step. Finally,conditioned on passing this last step, they perform privacyampliﬁcation on their keys to extract a secret key pair ( S
A,SB)
where |SA|=|SB|=/lscriptbits.
III. SECURITY BOUNDS
Before we state the security bounds for our protocol, it is
instructive to spell out the security criteria that we are using.For some small protocol errors, ε
cor,εsec>0, we say that our
protocol is εcor+εsecsecure if it is εcorcorrect and εsecsecret.
The former is satisﬁed if Pr[ SA/negationslash=SB]/lessorequalslantεcor, i.e., the secret
keys are identical except with a small probability εcor.T h e
latter is satisﬁed if (1 −pabort)/bardblρAE−UA⊗ρE/bardbl1/2/lessorequalslantεsecwhere ρAEis the classical-quantum state describing the joint
state of SAandE,UAis the uniform mixture of all possible
values of SA, andpabort is the probability that the protocol
aborts. Importantly, this secrecy criterion guarantees that theprotocol is universally composable: the pair of secret keys canbe safely used in any cryptographic task, e.g., for encryptingmessages, that requires a perfectly secure key [ 29].
In the following, we present only the necessary formulas
to compute the security bounds; the full security analysis isdeferred to Appendixes A and B.
The correctness of the protocol is guaranteed by the error-
veriﬁcation step. This step ensures that Bob’s corrected keyis identical to Alice’s key with probability at least 1 −ε
hash,
which implies that the ﬁnal secret keys ( SA,SB) are identical
with probability at least 1 −εhash. Therefore, the correctness
of the protocol is εcor=εhash.
Conditioned on passing the checks in the error-estimation
and error-veriﬁcation steps, a εsec-secret key of length
/lscript=/floorleftbigg
sX,0+sX,1−sX,1h(φX)
−λEC−6l o g221
εsec−log22
εcor/floorrightbigg
(1)
can be extracted, where h(x):=−xlog2x−(1−x)l o g2(1−
x) is the binary entropy function. Recall that sX,0,sX,1, and
φX=cX,1/sX,1are the number of vacuum events, the number
of single-photon events, and the phase error rate associatedwith the single-photons events in X
A, respectively. Next, we
show how to calculate them in two steps.
First, we extend the decoy-state analysis proposed in
Ref. [ 30] to the case of ﬁnite sample sizes. Accordingly, the
number of vacuum events in XAsatisﬁes
sX,0/greaterorequalslantτ0μ2n−
X,μ3−μ3n+
X,μ2
μ2−μ3, (2)
where τn:=/summationtext
k∈Ke−kknpk/n! is the probability that Alice
sends a n-photon state, and
n±
X,k:=ek
pk/bracketleftBigg
nX,k±/radicalBigg
nX
2ln21
εsec/bracketrightBigg
,∀k∈K.
The number of single-photon events in XAis
sX,1/greaterorequalslantτ1μ1/bracketleftbig
n−
X,μ2−n+
X,μ3−μ2
2−μ2
3
μ21/parenleftbig
n+
X,μ1−sX,0
τ0/parenrightbig/bracketrightbig
μ1(μ2−μ3)−μ2
2+μ2
3.(3)
We also calculate the number of vacuum events, sZ,0, and the
number of single-photon events, sZ,1,f o rZ=∪k∈KZk, i.e.,
by using Eqs. ( 2) and ( 3) with statistics from the basis Z.
In addition, the number of bit errors vZ,1associated with the
single-photon events in Zis also required. It is given by
vZ,1/lessorequalslantτ1m+
Z,μ2−m−
Z,μ3
μ2−μ3, (4)
where
m±
Z,k:=ek
pk/bracketleftBigg
mZ,k±/radicalBigg
mZ
2ln21
εsec/bracketrightBigg
,∀k∈K,
andmZ=/summationtext
k∈KmZ,k.
022307-2

CONCISE SECURITY BOUNDS FOR PRACTICAL DECOY- . . . PHYSICAL REVIEW A 89, 022307 (2014)
Second, the formula for the phase error rate of the single-
photon events in XAis [31]
φX:=cX,1
sX,1/lessorequalslantvZ,1
sZ,1+γ/parenleftbigg
εsec,vZ,1
sZ,1,sZ,1,sX,1/parenrightbigg
, (5)
where
γ(a,b,c,d ):=/radicalBigg
(c+d)(1−b)b
cdlog 2log2/parenleftbiggc+d
cd(1−b)b212
a2/parenrightbigg
.
IV . EVALUATION
We consider a ﬁber-based QKD system model that borrows
parameters from recent decoy-state QKD and single-photondetector experiments. In particular, we assume that Alicecan set the intensity of each laser pulse to one of the threepredetermined intensity levels, μ
1,μ2, and μ3=2×10−4
[32]. Bob uses an active measurement setup with two single-
photon detectors (InGaAs APDs): they have a detectionefﬁciency of η
Bob=10%, a dark count probability of pdc=
6×10−7, and an after-pulse probability of pap=4×10−2
[33]. The measurement has four possible outcomes {0,1,∅,⊥}
which correspond to bit values 0, 1, no detection, and doubledetection.
The system model is applied to two types of channel
architectures, namely one that uses a dedicated optical ﬁberfor the quantum channel and one that uses dense wavelengthdivision multiplexing (DWDM) to put the quantum channeltogether with the classical channels into one optical ﬁber (e.g.,see Refs. [ 34–36]). In both cases, we assume that the ﬁbers
have an attenuation coefﬁcient of 0 .2d B/km. That is, their
transmittance is η
ch=10−0.2L/10, where L(km) is the ﬁber
length.
The considered channel architectures, however, do not
have the same channel error model. For the dedicated ﬁber,the probability of having a bit error for intensity kis
e
k=pdc+emis[1−exp(−ηchk)]+papDk/2, where emisis
the error rate due to optical errors. Here, the expecteddetection rate (excluding after-pulse contributions) is D
k=
1−(1−2pdc)e x p (−ηsysk), where ηsys=ηchηBob.T h ee x -
pected detection rate (including after-pulse contributions) isthusR
k=Dk(1+ppa). The channel error model for the
DWDM architecture is more involved due to additional noise
contributions from Raman scattering and cross talks betweenchannels. We refer to Ref. [ 35] for details about it.
The parameter λ
ECis set to a simple function fECh(eobs)
where fECis the error-correction efﬁciency and eobsis the
average of the observed error rates in basis X(we note that
very recently, a more accurate theoretical model of λEChas
been derived in Ref. [ 37]). In practice, however, λECshould
be set to the size of the information exchanged during theerror-correction step. Regarding the secrecy, we set ε
secto be
proportional to the secret key length, that is, εsec=κ/lscriptwhere
κis a security constant; this security constant can be seen as
the secrecy leakage per generated bit.
For the evaluation, we numerically optimize the secret key
rateR:=/lscript/N over the free parameters {qx,pμ1,pμ2,μ1,μ2}
given that the set {κ,ε cor,emis,fEC,L,n X}is ﬁxed. Speciﬁcally,
we ﬁx κ=10−15,εcor=10−15,emis=5×10−3, andfEC=
1.16, and generate curves (see Fig. 1) for a range of realistic0 20 40 60 80 100 120 140 160 180 20010−710−610−510−410−310−2
Fiber length (km)Secret key rate per pulse
FIG. 1. (Color online) Secret key rate vs ﬁber length (dedicated
ﬁber). Numerically optimized secret key rates (in logarithmic scale)
are obtained for a ﬁxed postprocessing block size nX=10swith
s=4,5,..., 9 (from left to right). The dashed curve corresponds to
the asymptotic secret key rate, i.e., in the limit of inﬁnitely large keys;
however, here we still assume that the number of intensity levels is 3.The number of laser pulses sent by Alice can be approximated with
the secret key rate and the block size, i.e., N/lessorequalslantn
X/R.
postprocessing block sizes, i.e., nX=10swiths=4,5,..., 9.
From Fig. 1, we see that the security performances corre-
sponding to block sizes 107,108, and 109have only slight
differences. For example, at a ﬁber length of 100 km, thesecret key rate obtained with n
X=109is about 1 .75 times
the one based on nX=107. This suggests that it may not
be necessary to go to large block sizes (where computationalresources are high) to gain signiﬁcant improvements. On theother hand, for block sizes 10
4,105, and 106, there is a distinct
advantage in terms of the secret key rate and ﬁber length forlarger block sizes. This is expected since smaller block sizescorrespond to larger statistical ﬂuctuations in the estimationprocess. Interestingly, we see that even if we use a block sizeof 10
4, cryptographic keys can still be distributed over a ﬁber
length of 135 km. The same trend is observed for the DWMDchannel architecture (see Fig. 2).
V . CONCLUDING REMARKS
Although our security bounds are rather general and
can be applied to a wide class of implementations, someassumptions are still needed. In particular, we require thatthe probability of having a detection in Bob’s measurementdevice is independent of his basis choice. This assumption isnormally satisﬁed when the detectors are operating accordingto speciﬁcation. However, if the detectors are not implementedcorrectly, then there may be serious security consequences,e.g., see Ref. [ 39]; see also Ref. [ 40] for the corresponding
countermeasures. Alternatively, one can adopt the recentlyproposed measurement-device-independent QKD (mdiQKD)[41] to remove all detector side channels. We note, however,
that the implementation of mdiQKD is more complex than theone of decoy-state QKD, and the achievable ﬁnite-key secretkey rates are typically lower [ 42].
In summary, we have provided tight ﬁnite-key security
bounds for a practical decoy-state QKD protocol that can be
022307-3

LIM, CURTY , W ALENTA, XU, AND ZBINDEN PHYSICAL REVIEW A 89, 022307 (2014)
010 20 30 40 50 60 70 80 90100 110 120 13010−610−510−410−310−2
FIber length (km)Secret key rate per pulse
FIG. 2. (Color online) Secret key rate vs ﬁber length (DWDM).
We consider a (4+1) DWDM channel architecture [ 35] that puts four
classical channels and one quantum channel into an optical ﬁber. In
the simulation, we take that each classical channel has a power of −34
dBm at the receiver [ 38]. Numerically optimized secret key rates (in
logarithmic scale) are obtained for a ﬁxed postprocessing block size
nX=10swiths=4,5,..., 9 (from left to right). The dashed curve
corresponds to the asymptotic secret key rate, i.e., in the limit ofinﬁnitely long keys.
applied to existing QKD implementations. More importantly,
these bounds are secure against general attacks, and can beeasily computed by referring to just ﬁve concise formulas,i.e., Eqs. ( 1)–(5). On the application side, we also see that
secret keys can be securely distributed over large distanceswith rather small postprocessing block sizes. Accordingly,this allows existing QKD implementations to speed up theirkey-distillation processes.
ACKNOWLEDGMENTS
We thank M. Tomamichel and N. Gisin for helpful discus-
sions. We acknowledge support from the Swiss NCCR-QSIT,the NanoTera QCRYPT, the FP7 Marie-Curie IAAP QCERTproject, the European Regional Development Fund (ERDF),the Galician Regional Government (Projects No. CN2012/279and No. CN 2012/260, Consolidation of Research Units:AtlantTIC), NSERC, the CRC program, and the Paul BiringerGraduate Scholarship.
APPENDIX A: DECOY-STATE ANALYSIS
Here, we provide the details for the security bounds pre-
sented in the main text. The security analysis is a combinationof a proof technique based on entropic uncertainty relations[26] and a ﬁnite-size analysis for the two-decoy-state method.
In the following, we ﬁrst present the details for the decoy-stateanalysis.
Recall that our two-decoy-state method consists in Alice
setting the intensity of each laser pulse to one of thethree intensity levels, μ
1,μ 2, andμ3, where μ1>μ 2+μ3
andμ2>μ 3/greaterorequalslant0. Crucially, from the perspective of the
eavesdropper, the ﬁnal prepared state (i.e., with the encodedbit value) appears the same to her regardless of the choice ofintensity level (or equivalently, the average photon number).Therefore, one can imagine an equivalent counterfactualprotocol: one in which Alice has the ability to send n-photon
states, and she only decides on the choice of the average photonnumber after Bob has a detection. In the following, we providethe analysis for the Xbasis; the same analysis applies to the Z
basis.
Consider the case whereby Alice encodes the states in the X
basis and let s
X,nbe the number of detections observed by Bob
given that Alice sent n-photon states. Note that/summationtext∞
n=0sX,n=
nXis the total number of detections given that Alice sent states
prepared in the Xbasis. In the asymptotic limit, we expect nX,k
events from nXevents to be assigned to the intensity k, that is,
nX,k→n∗
X,k=∞/summationdisplay
n=0pk|nsX,n,∀k∈K={μ1,μ2,μ3},
where pk|nis the conditional probability of choosing the
intensity kgiven that Alice prepared a n-photon state. For ﬁnite
sample sizes, using Hoeffding’s inequality for independentevents [ 43], we have that n
X,ksatisﬁes
|n∗
X,k−nX,k|/lessorequalslantδ(nX,ε1), (A1)
with probability at least 1 −2ε1, where δ(nX,ε1):=√nX/2ln(1/ε1). Note that the deviation term δ(nX,ε1)i st h e
same for all values of k. Basically, Eq. ( A1) allows us to
establish a relation between the asymptotic values and the
observed statistics (i.e., nX,μ1,nX,μ2, and nX,μ3). Moreover,
the same relation can also be made for the expected numberof errors and the observed number of errors. Let v
X,nbe the
number of errors associated with sX,n, then in the asymptotic
limit, we expect mX,kerrors from mXerrors to be assigned to
the intensity k, i.e.,
mX,k→m∗
X,k=∞/summationdisplay
n=0pk|nvX,n,∀k∈K={μ1,μ2,μ3}.
Using Hoeffding’s inequality [ 43], we thus have for all values
ofk,
|m∗
X,k−mX,k|/lessorequalslantδ(mX,ε2), (A2)
which holds with probability at least 1 −2ε2.
For the moment, we keep these relations aside; they will
be needed later when we apply the decoy-state analysis (to bedetailed below) to the observed statistics.
1. Lower-bound on the number of vacuum events
An analytical lower-bound on sX,0can be established by
exploiting the structure of the conditional probabilities pk|n.
First of all, we note that with Bayes’ rule, for all k,w eh a v e
pk|n=pk
τnpn|k=pk
τne−kkn
n!, (A3)
where τn:=/summationtext
k∈Kpke−kkn/n! is the probability that Alice
prepares a n-photon state. Using this and following an
approach proposed by [ 30], we have that
μ2eμ3n∗
X,μ3
pμ3−μ3eμ2n∗
X,μ2
pμ2
=(μ2−μ3)sX,0
τ0−μ2μ3∞/summationdisplay
n=2/parenleftbig
μn−1
2−μn−1
3/parenrightbig
sX,n
n!τn,
022307-4

CONCISE SECURITY BOUNDS FOR PRACTICAL DECOY- . . . PHYSICAL REVIEW A 89, 022307 (2014)
where the second term on the right-hand side is non-negative
forμ2>μ 3. Rewriting the above expression for sX,0gives
sX,0/greaterorequalslantτ0
(μ2−μ3)/parenleftbiggμ2eμ3n∗
X,μ3
pμ3−μ3eμ2n∗
X,μ2
pμ2/parenrightbigg
. (A4)
Note that this lower bound is tight when μ3→0.
2. Lower bound on the number of single-photon events
The lower bound for the number of single-photon events
is slightly more involved, but it can be demonstrated in threeconcise steps.
First, note that
e
μ2n∗
X,μ2
pμ2−eμ3n∗
X,μ3
pμ3
=(μ2−μ3)sX,1
τ1+∞/summationdisplay
n=2/parenleftbig
μn
2−μn
3/parenrightbig
sX,n
n!τn
/lessorequalslant(μ2−μ3)sX,1
τ1+μ2
2−μ2
3
μ2
1∞/summationdisplay
n=2μn
1sX,n
n!τn,
where the inequality is due to
μn
2−μn
3=/parenleftbig
μ2
2−μ2
3/parenrightbig
(μ2+μ3)n−1/summationdisplay
i=0μn−i−1
2μi
3
/lessorequalslant/parenleftbig
μ2
2−μ2
3/parenrightbig
(μ2+μ3)n−2/lessorequalslant/parenleftbig
μ2
2−μ2
3/parenrightbig
μn−2
1,
forn/greaterorequalslant2 and μ2+μ3/lessorequalslantμ1. Note that we used/summationtextn−1
i=0μn−i−1
2μi
3/lessorequalslant(μ2+μ3)n−1forn/greaterorequalslant2.
Second, using the fact that the sum of multiphoton events
is given by
∞/summationdisplay
n=2μn
1sX,n
n!τn=eμ1n∗
X,μ1
pμ1−sX,0
τ0−μ1sX,1
τ1,
we further get
eμ2n∗
X,μ2
pμ2−eμ3n∗
X,μ3
pμ3/lessorequalslant(μ2−μ3)sX,1
τ1
+μ2
2−μ2
3
μ21/parenleftbiggeμ1n∗
X,μ1
pμ1−sX,0
τ0−μ1sX,1
τ1/parenrightbigg
.
Finally, solving for sX,1gives
sX,1/greaterorequalslantμ1τ1
μ1(μ2−μ3)−/parenleftbig
μ2
2−μ2
3/parenrightbig/bracketleftbiggeμ2n∗
X,μ2
pμ2
−eμ3n∗
X,μ3
pμ3+μ2
2−μ2
3
μ21/parenleftbiggsX,0
τ0−eμ1n∗
X,μ1
pμ1/parenrightbigg/bracketrightbigg
.(A5)
3. Upper bound on the number of single-photon errors
An upperbound on the number of single-photon errors
can be obtained with just m∗
X,μ2andm∗
X,μ3; i.e., by taking
eμ2m∗
X,μ2/pμ2−eμ3m∗
X,μ3/pμ3, it is easy to show that
vX,1/lessorequalslantτ1
μ2−μ3/parenleftbiggeμ2m∗
X,μ2
pμ2−eμ3m∗
X,μ3
pμ3/parenrightbigg
. (A6)4. Finite-size decoy-state analysis
The bounds given above are still not applicable to the observed
statistics since Eqs. ( A4)–(A6) involve terms that are valid
only in the asymptotic limit, i.e., {n∗
X,k}k∈Kand{m∗
X,k}k∈K.
However, this is easily resolved by using Eqs. ( A1) and ( A2).
Speciﬁcally, let
n∗
X,k/lessorequalslantnX,k+δ(nX,ε1)=:˜n+
X,k, (A7)
n∗
X,k/greaterorequalslantnX,k−δ(nX,ε1)=:˜n−
X,k, (A8)
and
m∗
X,k/lessorequalslantmX,k+δ(mX,ε2)=:˜m+
X,k, (A9)
m∗
X,k/greaterorequalslantmX,k−δ(mX,ε2)=:˜m−
X,k(A10)
for all values of k. Putting them into Eqs. ( A4)–(A6), we thus
have the formulas as stated in the main text.
APPENDIX B: SECRECY ANALYSIS
The secrecy analysis roughly follows along the lines of
Ref. [ 26], i.e., we use a certain family of entropic uncertainty
relations to establish bounds on the smooth min-entropy of theraw key conditioned on Eve’s information.
To start with, let system E
/primebe the information that Eve
gathers on XA, i.e., the raw key of Alice, up to the error-
veriﬁcation step. By applying privacy ampliﬁcation with two-universal hashing [ 29], aε
sec-secret key of length /lscriptcan be
extracted from XA. Speciﬁcally, the secret key is εsecsecret if
/lscriptis chosen such that
/lscript=/floorleftbigg
Hν
min(XA|E/prime)−2l o g21
2ν/floorrightbigg
, (B1)
forν+ν/lessorequalslantεsecwhere ν,νare chosen to be proportional to
εsec/(1−pabort). Here, Hν
min(XA|E/prime) is the conditional smooth
min-entropy, which quantiﬁes the amount of uncertaintysystem E
/primehas on XA. In fact, this quantity is the heart of
our security analysis. In the following, we show how to boundH
ν
min(XA|E/prime) using statistics obtained in the protocol.
First, using a chain-rule inequality for smooth entropies,
and the fact that λECbits and log22/εcorbits of in-
formation were published during the error-correction anderror-veriﬁcation steps, respectively, we get H
ν
min(XA|E/prime)/greaterorequalslant
Hν
min(XA|E)−λEC−log22/εcor, where system Eis the
remaining (possibly quantum) information Eve has onX
A. In general, λEC should be determined by the
amount of leakage the actual protocol reveals during theerror-correction step.
Second, we decompose X
AintoXv
AXs
AXm
A, which are the
corresponding bit strings due to the vacuum, single-photon,and multiphoton events. Note that this decomposition is knownto Eve, i.e., the decomposition information is included insidesystem E. By using a generalized chain-rule result from
Ref. [ 44], we have that
H
ν
min(XA|E)/greaterorequalslantHα1
min/parenleftbig
Xs
A|Xv
AXmAE/parenrightbig
+Hα3+2α4+α5
min/parenleftbig
Xv
AXmA|E/parenrightbig
−2l o g21
α2−1,
022307-5

LIM, CURTY , W ALENTA, XU, AND ZBINDEN PHYSICAL REVIEW A 89, 022307 (2014)
forν=2α1+α2+(α3+2α4+α5) where αi>0 for all i.
Next, we use the same chain rule again on the second term onthe right-hand side to get
H
α3+2α4+α5
min/parenleftbig
Xv
AXmA|E/parenrightbig
/greaterorequalslantHα4
min/parenleftbig
Xm
A|Xv
AE/parenrightbig
+Hα5
min/parenleftbig
Xv
A|E/parenrightbig
−2l o g21
α3−1
/greaterorequalslantsX,0−2l o g21
α3−1.
To get the second inequality, we used Hα4min(Xm
A|Xv
AE)/greaterorequalslant
0 and Hα5min(Xv
A|E)/greaterorequalslantHmin(Xv
A|E)=Hmin(Xv
A)=log22sX,0=
sX,0. The former is given by the fact that all multiphoton
events are taken to be insecure, i.e., due to the photon-number-splitting attack. The latter is based on the assumptionthat vacuum contributions contain zero information about thechosen bit values and the bits are uniformly distributed.
Third, we provide a bound on the remaining smooth min-
entropy quantity which is now restricted to the single-photonevents, i.e., via the uncertainty relation for smooth entropies[25]. Under the assumption that Alice prepares the states using
mutually unbiased bases (i.e., Xis the computational basis and
Zis the Hadamard basis), we can further bound this quantity
with the max-entropy between Alice and Bob, which is directlygiven by the amount of correlation between them [ 26]. More
precisely, we have
H
α1
min/parenleftbig
Xs
A|Xv
AXmAE/parenrightbig
/greaterorequalslantsX,1−Hα1
max/parenleftbig
Zs
A|Zs
B/parenrightbig
/greaterorequalslantsX,1/bracketleftbigg
1−h/parenleftbiggcX,1
sX,1/parenrightbigg/bracketrightbigg
,
where the ﬁrst inequality is given by the uncertainty relation
[25] and the smooth max-entropy Hα1max(Zs
A|Zs
B) is a measure
of correlations between Zs
AandZs
B. Here, Zs
AandZs
Barethe bit strings Alice and Bob would have obtained if they
had measured in the basis Zinstead. The second inequality is
achieved by using Hα1max(Zs
A|Zs
B)/lessorequalslantsX,1h(cX,1/sX,1)( s e e[ 26,
Lemma 3]), where cX,1is the number of phase errors in the
single-photon events. Here, the number of phase errors cX,1
has to be estimated via a random-sampling theory (without
replacement) as these errors are not directly observed inthe protocol. More concretely, by using a random samplingwithout replacement result given in Ref. [ 31] which is
based on an approximation technique for the hypergeometricdistribution, we have with probability at least 1 −α
1,
cX,1
sX,1/lessorequalslantvZ,1
sZ,1+γ/parenleftbigg
α1,vZ,1
sZ,1,sZ,1,sX,1/parenrightbigg
, (B2)
where
γ(a,b,c,d ):=/radicalBigg
(c+d)(1−b)b
cdlog 2log2/parenleftbiggc+d
cd(1−b)b1
a2/parenrightbigg
.
Fourth, putting everything together, we arrive at a secret
key length of
/lscript=/floorleftbigg
sX,0+sX,1/bracketleftbigg
1−h/parenleftbiggcX,1
sX,1/parenrightbigg/bracketrightbigg
−λEC−log22
εcorβ/floorrightbigg
,
(B3)
where β:=(α2α3ν)2. Note that sX,0,sX,1,sZ,0,sZ,1,vZ,1are
to be bounded by Eqs. ( A4)–(A6) using the relations given by
Eqs. ( A7)–(A10).
Finally, after composing the error terms due to ﬁnite-sample
sizes and setting α4=α5=0, the secrecy is
εsec=2[2α1+α2+α3]+ν+10ε1+2ε2. (B4)
To get the secrecy given in the main text we set each error term
to a common value ε, thusεsec=21ε.
[1] C. H. Bennett and G. Brassard, in Proceedings of IEEE
International Conference on Computers, Systems, and SignalProcessing, Bangalore, India (IEEE, New York, 1984),
pp. 175–179.
[2] N. Gisin, G. Ribordy, W. Tittel, and H. Zbinden, Rev. Mod.
Phys. 74,145 (2002 ); V . Scarani, H. Bechmann-Pasquinucci,
N. J. Cerf, M. Du ˇsek, N. L ¨utkenhaus, and M. Peev, ibid. 81,
1301 (2009 ).
[3] B. Huttner, N. Imoto, N. Gisin, and T. Mor, Phys. Rev. A 51,
1863 (1995 ); G. Brassard, N. L ¨utkenhaus, T. Mor, and B. C.
Sanders, P h y s .R e v .L e t t . 85,1330 (2000 ).
[4] Y . Zhao, B. Qi, X. Ma, H. K. Lo, and L. Qian, Phys. Rev. Lett.
96,070502 (2006 ).
[5] D. Rosenberg, J. W. Harrington, P. R. Rice, P. A. Hiskett, C.
G. Peterson, R. J. Hughes, A. E. Lita, S. W. Nam, and J. E.Nordholt, Phys. Rev. Lett. 98,010503 (2007 ).
[6] C. Z. Peng, J. Zhang, D. Yang, W. B. Gao, H. X. Ma, H. Yin, H.
P. Zeng, T. Yang, X. B. Wang, and J. W. Pan, Phys. Rev. Lett.
98,010505 (2007 ).
[7] Z. L. Yuan, A. W. Sharpe, and A. J. Shields, Appl. Phys. Lett.
90,011118 (2007 ).
[ 8 ] A .R .D i x o n et al. ,Opt. Express 16,18790 (2008 ).[9] A. Tanaka et al. ,Opt. Express 16,11354 (2008 ).
[10] D. Rosenberg et al. ,
New J. Phys. 11,045009
(2009 ).
[11] A. R. Dixon, Z. L. Yuan, J. F. Dynes, A. W. Sharpe, and A. J.
Shields, Appl. Phys. Lett. 96,161102 (2010 ).
[12] Y . Liu et al. ,Opt. Express 18,8587 (2010 ).
[13] M. Sasaki et al. ,Opt. Express 19,10387 (2011 ).
[14] J. Y . Wang et al. ,Nat. Photon. 7,387(2013 ).
[15] W. Y . Hwang, P h y s .R e v .L e t t . 91,057901 (2003 ).
[16] H.-K. Lo, X. Ma, and K. Chen, Phys. Rev. Lett. 94,230504
(2005 ).
[17] X. B. Wang, P h y s .R e v .L e t t . 94,230503 (2005 ).
[18] M. Hayashi, Phys. Rev. A 76,012329 (2007 ); J. Hasegawa,
M. Hayashi, T. Hiroshima, and A. Tomita, arXiv:0707.3541 .
[19] R. Cai and V . Scarani, New J. Phys. 11,045024 (2009 ).
[20] T. T. Song, J. Zhang, S. J. Qin, and Q. Y . Wen, Quantum Inf.
Comput. 11, 374 (2011).
[21] R. D. Somma and R. J. Hughes, Phys. Rev. A 87,062330
(2013 ).
[22] Z. Wei, W. Wang, Z. Zhang, M. Gao, Z. Ma, and X. Ma, Sci.
Rep.3,2453 (2013 ).
[23] M. Lucamarini et al. ,Opt. Express 21,24550 (2013 ).
022307-6

CONCISE SECURITY BOUNDS FOR PRACTICAL DECOY- . . . PHYSICAL REVIEW A 89, 022307 (2014)
[24] M. Hayashi and R. Nakayama, arXiv:1302.4139v4 .
[25] M. Tomamichel and R. Renner, Phys. Rev. Lett. 106,110506
(2011 ).
[26] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, Nat.
Commun. 3,634(2012 ).
[27] H.-K. Lo, H. F. Chau, and M. Ardehali, J. Cryptol. 18,133
(2005 ).
[28] M. N. Wegman and J. L. Carter, J. Comput. Syst. Sci. 22,265
(1981 ).
[29] R. Renner, arXiv:quant-ph/0512258 .
[30] X. Ma, B. Qi, Y . Zhao, and H.-K. Lo, P h y s .R e v .A 72,012326
(2005 ).
[31] C. H. F. Fung, X. Ma, and H. F. Chau, Phys. Rev. A 81,012318
(2010 ).
[32] B. Fr ¨ohlich et al. ,Nature (London) 501,69(2013 ).
[33] N. Walenta et al. ,J. Appl. Phys. 112,063106 (2012 ).
[34] N. A. Peters et al. ,New J. Phys. 11,045012 (2009 ).
[35] P. Eraerds et al. ,New J. Phys. 12,063027 (2010 ).[36] K. A. Patel, J. F. Dynes, I. Choi, A. W. Sharpe, A. R. Dixon,
Z. L. Yuan, R. V . Penty, and A. J. Shields, P h y s .R e v .X 2,
041010 (2012 ).
[37] M. Tomamichel, J. Martinez-Mateo, C. Pacher, and D. Elkouss,
arXiv:1401.5194 .
[38] We remark that −34 dBm is achievable with commercial
devices, e.g., see OptiCin SFP-DWDM-15xx.xx-28. Note that inRef. [ 35], they worked with a receiver sensitivity of −28 dBm.
[39] L. Lydersen et al. ,Nat. Photon. 4,686(2010 ).
[40] Z. L. Yuan, J. F. Dynes, and A. J. Shields, Appl. Phys. Lett.
98,231104 (2011 ); M. Legr ´e and G. Robordy, intl. patent. appl.
WO 2012/046135 A2 (ﬁled in 2010).
[41] H.-K. Lo, M. Curty, and B. Qi, Phys. Rev. Lett. 108,130503
(2012 ).
[42] M. Curty et al. ,arXiv:1307.1081 .
[43] W. Hoeffding, J. Amer. Stat. Assoc. 58,13(1963 ).
[44] A. Vitanov, F. Dupuis, M. Tomamichel, and R. Renner, IEEE
Trans. Inf. Theor. 59,2603 (2013 ).
022307-7
