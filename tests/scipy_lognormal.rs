// This file is created by gen_scipy_tests.py and contains reference
// tests whose expected values are produced by SciPy at generation time.
//

// Each test compares our kernel outputs against SciPy with a per-test tolerance.
// NaN/Inf equality is handled by util::assert_slice_close.

mod util;
#[cfg(feature = "probability_distributions")]
mod scipy_lognormal_tests {
    use super::util::assert_slice_close;
    use minarrow::vec64;
    use simd_kernels::kernels::scientific::distributions::univariate::lognormal::{
        lognormal_cdf, lognormal_pdf, lognormal_quantile,
    };
    // TODO[5]: Fix decimal points as source
    // Re-running the Scipy file generation will ruin this file

    #[test]
    fn lognormal_pdf_standard() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.28159018901526833,
            0.62749607711592448,
            0.3989422804014327,
            0.15687401927898112,
            0.021850714830327203,
            0.0028159018901526803,
            0.00022444570374742122,
            3.7908376929382674e-06
        ];
        let got = lognormal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_shifted_mean() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.017079308311203585,
            0.19029780481010564,
            0.24197072451914337,
            0.19029780481010555,
            0.066265642406154915,
            0.017079308311203578,
            0.0027226640152717943,
            0.00011496296433806829
        ];
        let got = lognormal_pdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_small_sigma() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.00019804773299183628,
            0.61045530419018312,
            0.79788456080286541,
            0.15261382604754578,
            0.00089757843107166459,
            1.9804773299183516e-06,
            6.394917882875848e-10,
            8.1311237759679834e-16
        ];
        let got = lognormal_pdf(&x, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_large_sigma() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            1.0281510740412525,
            0.37568841601677122,
            0.19947114020071638,
            0.093922104004192791,
            0.028859676775298194,
            0.010281510740412525,
            0.0032483130817288646,
            0.0005889912567987218
        ];
        let got = lognormal_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_both_varied() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.46428380724168228,
            0.38766564152452521,
            0.25158881846199549,
            0.13188288209943969,
            0.040463120388184191,
            0.01291897273132863,
            0.0033316522298316759,
            0.00040021215837279063
        ];
        let got = lognormal_pdf(&x, 0.5, 1.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_deterministic() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            2.9624799253490976e-114,
            2.9446618075981229e-10,
            3.9894228040143269,
            7.3616545189953319e-11,
            4.5133480928065897e-57,
            2.9624799253488177e-116,
            2.6485167661646355e-196,
            0.0
        ];
        let got = lognormal_pdf(&x, 0.0, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_high_variability() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.9905295063827837,
            0.2589564248086107,
            0.13298076013381091,
            0.06473910620215266,
            0.023031469567115919,
            0.0099052950638278316,
            0.004038582431782024,
            0.0011365114646704345
        ];
        let got = lognormal_pdf(&x, 0.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_extended() {
        let x = vec64![
            9.9999999999999995e-07,
            0.01,
            0.10000000000000001,
            1.0,
            10.0,
            100.0,
            1000.0
        ];
        let expect = vec64![
            1.4268502377012741e-36,
            0.00099023866495918152,
            0.28159018901526833,
            0.3989422804014327,
            0.0028159018901526803,
            9.9023866495917625e-08,
            1.7349107871679253e-14
        ];
        let got = lognormal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_standard_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.010651099341700129,
            0.24410859578558275,
            0.5,
            0.75589140421441725,
            0.94623968954833682,
            0.98934890065829983,
            0.99863106651214195,
            0.99995423690475016
        ];
        let got = lognormal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_shifted_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.00047899008636511357,
            0.045213727790224138,
            0.15865525393145707,
            0.37947770112008489,
            0.72888289265311779,
            0.90364177511826438,
            0.97701846391245784,
            0.99820451906570351
        ];
        let got = lognormal_cdf(&x, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_small_sigma_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            2.0606433959717227e-06,
            0.082828519001698464,
            0.5,
            0.91717148099830159,
            0.99935652898708616,
            0.99999793935660408,
            0.99999999896020164,
            0.99999999999999745
        ];
        let got = lognormal_cdf(&x, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_large_sigma_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.12480595124085969,
            0.36445584473653569,
            0.5,
            0.63554415526346431,
            0.78950906095123674,
            0.87519404875914031,
            0.93291598332860215,
            0.97476810045546813
        ];
        let got = lognormal_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_both_varied_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.030853860643602894,
            0.21318128471793008,
            0.36944134018176367,
            0.55122811530639426,
            0.77023629951114569,
            0.8852646431412956,
            0.95192594852453383,
            0.98853749358398224
        ];
        let got = lognormal_cdf(&x, 0.5, 1.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_deterministic_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            1.2841756306435786e-117,
            2.0824223487002437e-12,
            0.5,
            0.99999999999791755,
            1.0,
            1.0,
            1.0,
            1.0
        ];
        let got = lognormal_cdf(&x, 0.0, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_high_var_cdf() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let expect = vec64![
            0.22138371779204508,
            0.40863834423098205,
            0.5,
            0.59136165576901789,
            0.70418633224265159,
            0.77861628220795498,
            0.84100027960794355,
            0.90388451611678133
        ];
        let got = lognormal_cdf(&x, 0.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_cdf_extended_cdf() {
        let x = vec64![
            9.9999999999999995e-07,
            0.01,
            0.10000000000000001,
            1.0,
            10.0,
            100.0,
            1000.0
        ];
        let expect = vec64![
            1.0274605390204012e-43,
            2.0606433959717227e-06,
            0.010651099341700129,
            0.5,
            0.98934890065829983,
            0.99999793935660408,
            0.99999999999753808
        ];
        let got = lognormal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_standard_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.097651733070335991,
            0.19304081669873652,
            0.27760624185200983,
            0.50941628386327753,
            1.0,
            1.963031084158257,
            3.6022244792791573,
            5.180251602233013,
            10.240473656312131
        ];
        let got = lognormal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_shifted_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.26544493152262755,
            0.52473934418306889,
            0.75461200269312512,
            1.3847370275466819,
            2.7182818284590451,
            5.3360717247676481,
            9.7918613440548796,
            14.081383797195853,
            27.836493454766824
        ];
        let got = lognormal_quantile(&q, 1.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_small_sigma_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.31249277282896637,
            0.43936410492749239,
            0.52688351829603652,
            0.713734042808158,
            1.0,
            1.4010821118543542,
            1.8979527073347104,
            2.276016608514317,
            3.2000740079429622
        ];
        let got = lognormal_quantile(&q, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_large_sigma_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.0095358609716401522,
            0.037264756911715193,
            0.077065225515196581,
            0.25950495026507131,
            1.0,
            3.8534910373715419,
            12.976021199117996,
            26.835006662437703,
            104.86730070562277
        ];
        let got = lognormal_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_both_varied_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.050311489369987021,
            0.13983663362293491,
            0.2411521183071379,
            0.59945484689316209,
            1.6487212707001282,
            4.534589790285759,
            11.272062827152808,
            19.438982175363325,
            54.029047092384786
        ];
        let got = lognormal_quantile(&q, 0.5, 1.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_deterministic_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.79244293082251371,
            0.84833017412083656,
            0.87971687471595794,
            0.93477541629646888,
            1.0,
            1.0697756729225374,
            1.136729360026064,
            1.1787863151706772,
            1.2619205258882846
        ];
        let got = lognormal_quantile(&q, 0.0, 0.10000000000000001, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_ppf_high_var_ppf() {
        let q = vec64![
            0.01,
            0.050000000000000003,
            0.10000000000000001,
            0.25,
            0.5,
            0.75,
            0.90000000000000002,
            0.94999999999999996,
            0.98999999999999999
        ];
        let expect = vec64![
            0.00093119335019843901,
            0.0071936191083173916,
            0.021393787632751342,
            0.13219604740815724,
            1.0,
            7.5645226888855852,
            46.742541207108133,
            139.01208625902646,
            1073.8908302844925
        ];
        let got = lognormal_quantile(&q, 0.0, 3.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1.0000000000000001e-15);
    }

    #[test]
    fn lognormal_pdf_lognorm_low_var() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.00019804773299183628,
            0.61045530419018312,
            0.79788456080286541,
            0.15261382604754578,
            0.00089757843107166459
        ];
        let got = lognormal_pdf(&x, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_pdf_lognorm_standard() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.28159018901526833,
            0.62749607711592448,
            0.3989422804014327,
            0.15687401927898112,
            0.021850714830327203
        ];
        let got = lognormal_pdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_pdf_lognorm_high_var() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            1.0281510740412525,
            0.37568841601677122,
            0.19947114020071638,
            0.093922104004192791,
            0.028859676775298194
        ];
        let got = lognormal_pdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_pdf_lognorm_scaled() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            1.2789835765751681e-07,
            0.034174747548216075,
            0.30522765209509156,
            0.3989422804014327,
            0.029765458285142866
        ];
        let got = lognormal_pdf(&x, 0.69314718055994529, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_cdf_lognorm_cdf_low_var() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            2.0606433959717227e-06,
            0.082828519001698464,
            0.5,
            0.91717148099830159,
            0.99935652898708616
        ];
        let got = lognormal_cdf(&x, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_cdf_lognorm_cdf_standard() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.010651099341700129,
            0.24410859578558275,
            0.5,
            0.75589140421441725,
            0.94623968954833682
        ];
        let got = lognormal_cdf(&x, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_cdf_lognorm_cdf_high_var() {
        let x = vec64![0.10000000000000001, 0.5, 1.0, 2.0, 5.0];
        let expect = vec64![
            0.12480595124085969,
            0.36445584473653569,
            0.5,
            0.63554415526346431,
            0.78950906095123674
        ];
        let got = lognormal_cdf(&x, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_ppf_lognorm_ppf_low_var() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            0.31249277282896637,
            0.52688351829603652,
            1.0,
            1.8979527073347104,
            3.2000740079429622
        ];
        let got = lognormal_quantile(&q, 0.0, 0.5, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_ppf_lognorm_ppf_standard() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            0.097651733070335991,
            0.27760624185200983,
            1.0,
            3.6022244792791573,
            10.240473656312131
        ];
        let got = lognormal_quantile(&q, 0.0, 1.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }

    #[test]
    fn lognormal_ppf_lognorm_ppf_high_var() {
        let q = vec64![
            0.01,
            0.10000000000000001,
            0.5,
            0.90000000000000002,
            0.98999999999999999
        ];
        let expect = vec64![
            0.0095358609716401522,
            0.077065225515196581,
            1.0,
            12.976021199117996,
            104.86730070562277
        ];
        let got = lognormal_quantile(&q, 0.0, 2.0, None, None).unwrap();
        assert_slice_close(&got, &expect, 1e-15);
    }
}
