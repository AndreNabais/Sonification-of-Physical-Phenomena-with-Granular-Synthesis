s.boot;


b = Buffer.readChannel(s, "C:/Users/andre/Downloads/663482__sergequadrado__birds-sound.wav", channels:[1]);
b.play;



(
SynthDef(\grainCloud, {
	arg bufnum, dur=0.005, spread=2, amp=2.2, rate=10, pos=0.2, pan=0, jitter=0.01;
	var sig, trig, position;

	trig = Impulse.ar(rate);  // Create a trigger signal: impulses at the given rate
	position = pos + LFNoise1.ar(100).bipolar(jitter); //generates smooth random values at 100 Hz, scales them between -jitter and +jitter
	position = position.clip(0.0, 1.0); //position range is within the buffer duration

	sig = GrainBuf.ar(
		1,  //number of channels (1 = mono grain)
		trig,
		dur,
		bufnum,
		1,  //rate scale (1 = normal playback rate)
		position,
		spread,
		LFNoise1.kr(2),  // random pitch variation (used as interpolation here)
		-1,  // pan position
		512  // envelope size (number of samples)
	) * amp;  // scale grain signal by amplitude

// Output the audio signal with stereo panning
	Out.ar(0, Pan2.ar(sig, pan));
}).add;
)


//Instance creation
~g = Synth(\grainCloud, [\bufnum, b])


// OSC receivers
OSCdef(\rate,    { |msg| ~g.set(\rate, msg[1]) },    "/grain/rate");
OSCdef(\jitter,    { |msg| ~g.set(\jitter, msg[1]) },    "/grain/jitter");
OSCdef(\pos,     { |msg| ~g.set(\pos, msg[1]) },     "/grain/pos");
OSCdef(\spread,  { |msg| ~g.set(\spread, msg[1]) },  "/grain/spread");
OSCdef(\amp,     { |msg| ~g.set(\amp, msg[1]) },     "/grain/amp");
OSCdef(\pan,     { |msg| ~g.set(\pan, msg[1]) },     "/grain/pan");
OSCdef(\stop, {
    ~g.free;
    "Grain synth stopped.".postln;
}, "/grain/stop");

//record
s.makeWindow;