import QRCode from "qrcode";

export class CertificateService {
  async generate(input: { studentName: string; courseName: string; passedFinalAssessment: boolean; completion: number }) {
    if (input.completion < 100 || !input.passedFinalAssessment) throw new Error("Certificate requirements not met");
    const certificateId = `CMAI-${Date.now()}-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;
    const verificationUrl = `https://codementor.ai/verify/${certificateId}`;
    return {
      certificateId,
      studentName: input.studentName,
      courseName: input.courseName,
      completionDate: new Date().toISOString(),
      qrCode: await QRCode.toDataURL(verificationUrl),
      verificationUrl
    };
  }
}
