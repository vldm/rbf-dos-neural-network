use nalgebra::DVector;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct KDDRecord {
    duration: u32,
    protocol_type: String,
    service: String,
    flag: String,
    src_bytes: u32,
    dst_bytes: u32,
    land: String,
    wrong_fragment: u32,
    urgent: u32,
    hot: u32,
    num_failed_logins: u32,
    logged_in: String,
    num_compromised: u32,
    root_shell: u32,
    su_attempted: u32,
    num_root: u32,
    num_file_creations: u32,
    num_shells: u32,
    num_access_files: u32,
    num_outbound_cmds: u32,
    is_host_login: String,
    is_guest_login: String,
    count: u32,
    srv_count: u32,
    serror_rate: f64,
    srv_serror_rate: f64,
    rerror_rate: f64,
    srv_rerror_rate: f64,
    same_srv_rate: f64,
    diff_srv_rate: f64,
    srv_diff_host_rate: f64,
    dst_host_count: f64,
    dst_host_srv_count: f64,
    dst_host_same_srv_rate: f64,
    dst_host_diff_srv_rate: f64,
    dst_host_same_src_port_rate: f64,
    dst_host_srv_diff_host_rate: f64,
    dst_host_serror_rate: f64,
    dst_host_srv_serror_rate: f64,
    dst_host_rerror_rate: f64,
    dst_host_srv_rerror_rate: f64,
}

impl KDDRecord {
    pub fn input_vector(&self) -> DVector<f64> {
        DVector::from_row_slice(&[
            self.duration.into(),
            self.src_bytes.into(),
            self.dst_bytes.into(),
            self.urgent.into(),
            self.hot.into(),
            self.su_attempted.into(),
            self.root_shell.into(),
            self.num_failed_logins.into(),
            if self.is_host_login == "1" {1.} else {0.} ,
            if self.is_guest_login == "1" {1.} else {0.} ,
            self.count.into(),
            self.srv_count.into(),
            self.serror_rate,
            self.srv_serror_rate,
            self.rerror_rate,
            self.srv_rerror_rate,
            self.same_srv_rate,
            self.diff_srv_rate,
            self.srv_diff_host_rate,
            self.dst_host_count,
            self.dst_host_srv_count,
            self.dst_host_same_srv_rate,
            self.dst_host_diff_srv_rate,
            self.dst_host_same_src_port_rate,
            self.dst_host_srv_diff_host_rate,
            self.dst_host_serror_rate,
            self.dst_host_srv_serror_rate,
            self.dst_host_rerror_rate,
            self.dst_host_srv_rerror_rate,
        ])
    }
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types)]
enum Label {
    back,
    buffer_overflow,
    ftp_write,
    guess_passwd,
    imap,
    ipsweep,
    land,
    loadmodule,
    multihop,
    neptune,
    nmap,
    normal,
    perl,
    phf,
    pod,
    portsweep,
    rootkit,
    satan,
    smurf,
    spy,
    teardrop,
    warezclient,
    warezmaster,


}

// back dos
        // buffer_overflow u2r
        // ftp_write r2l
        // guess_passwd r2l
        // imap r2l
        // ipsweep probe
// land dos
        // loadmodule u2r
        // multihop r2l
// neptune dos
        // nmap probe
        // perl u2r
        // phf r2l
// pod dos
        // portsweep probe
        // rootkit u2r
        // satan probe
// smurf dos
        // spy r2l
// teardrop dos
        // warezclient r2l
        // warezmaster r2l

#[derive(Debug, Deserialize)]
pub struct KDDRecordWithLable {
    record: KDDRecord,
    label: Label,
}
impl KDDRecordWithLable {
    pub fn is_dos(&self) -> bool {
        match self.label {
            Label::back | Label::land 
             | Label::neptune | Label::pod 
             | Label::smurf | Label::teardrop => true,
            _ => false
        }
    }


   
    pub fn input_vector(&self) -> DVector<f64> {
        self.record.input_vector()

    }

    pub fn output_vector(&self) -> DVector<f64> {
        // DVector::from_row_slice(&[if self.is_dos() {
        //     1.
        // } else {
        //     0.
        // }])
        match self.label {
            Label::back =>      DVector::from_row_slice(&[1., 0., 0., 0., 0., 0., 0., 0.]),
            Label::land =>      DVector::from_row_slice(&[0., 1., 0., 0., 0., 0., 0., 0.]),
            Label::neptune =>   DVector::from_row_slice(&[0., 0., 1., 0., 0., 0., 0., 0.]),
            Label::pod =>       DVector::from_row_slice(&[0., 0., 0., 0., 1., 0., 0., 0.]),
            Label::smurf =>     DVector::from_row_slice(&[0., 0., 0., 0., 0., 1., 0., 0.]),
            Label::teardrop =>  DVector::from_row_slice(&[0., 0., 0., 0., 0., 0., 1., 0.]),
            _ =>                DVector::from_row_slice(&[0., 0., 0., 0., 0., 0., 0., 1.]),
        }
    }
}