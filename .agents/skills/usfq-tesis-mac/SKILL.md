---
name: usfq-tesis-mac
description: Conectar una Mac al servidor USFQ de la tesis Mesa y Memoria por SSH, abrir el proyecto en VS Code y acceder al chat mediante el puerto 8767.
---

# Mesa y Memoria desde la Mac

Conecta la Mac a la red interna o a la VPN de la USFQ. El modelo corre en la A100 del servidor. Si `ssh servidor-tesis` ya funciona en la Mac, ve directamente a **Abrir el chat**.

## Preparar la llave de la Mac una sola vez

1. En Terminal de la **Mac**, ejecuta estos comandos. Si ya tienes una llave con ese nombre, conserva la existente. El último comando copia la parte pública al portapapeles.

   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   if [ ! -e ~/.ssh/id_ed25519_usfq_mac ] && [ ! -e ~/.ssh/id_ed25519_usfq_mac.pub ]; then
     ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_usfq_mac -C "Mac USFQ"
   fi
   pbcopy < ~/.ssh/id_ed25519_usfq_mac.pub
   ```

2. Pasa la línea pública que empieza por `ssh-ed25519` al **equipo que ya tiene acceso**. Desde ese equipo, conéctate por SSH con tu configuración habitual y luego `nano ~/.ssh/authorized_keys`. Añade la línea pública al final. Guarda con **Ctrl+O**, pulsa **Enter** y sal con **Ctrl+X**. La llave privada se queda en la Mac.

3. De vuelta en la **Mac**, abre `nano ~/.ssh/config`. Sustituye `HOST_DEL_SERVIDOR` y `TU_USUARIO` por los datos que te facilitó la USFQ y añade el bloque. Si ya existe una entrada para el mismo alias, actualízala.

   ```sshconfig
   Host servidor-tesis
       HostName HOST_DEL_SERVIDOR
       User TU_USUARIO
       Port 22
       IdentityFile ~/.ssh/id_ed25519_usfq_mac
       IdentitiesOnly yes
   ```

4. Guarda el archivo y ejecuta `chmod 600 ~/.ssh/config` y `ssh servidor-tesis`. En la primera conexión, compara la huella ED25519 con la que te facilitó la USFQ por un canal de confianza antes de aceptar. Para volver a la terminal de la Mac, escribe `exit`.

## Abrir el chat

**Con VS Code.** Instala la extensión **Remote - SSH** de Microsoft. Pulsa **⌘⇧P**, elige **Remote-SSH: Connect to Host…** y selecciona **servidor-tesis**. En **Abrir carpeta**, introduce `~/tesis_clarificacion_longitudinal`. En la pestaña **Puertos**, elige **Reenviar un puerto** e introduce `8767`. Abre la dirección local que muestre VS Code y añade `/chat`. La simulación guiada está en `/`.

**Solo con Terminal.** Si prefieres abrir el chat directamente, ejecuta esto en una terminal **local de la Mac** y déjala abierta mientras usas el chat.

```bash
ssh -o ExitOnForwardFailure=yes -N -L 127.0.0.1:8767:127.0.0.1:8767 servidor-tesis
```

Abre el [chat de alimentación](http://127.0.0.1:8767/chat). Para cerrar el túnel, pulsa **Ctrl+C**. Si el puerto local ya está ocupado, cambia el primer `8767` por `18767` y abre `http://127.0.0.1:18767/chat`.

Si SSH funciona pero el chat no abre, comprueba el backend desde la Mac:

```bash
ssh servidor-tesis 'curl -fsS http://127.0.0.1:8767/api/timing/catalog'
```

Si recibes un JSON con SFT y GRPO, el backend responde. Revisa la dirección local en **Puertos**. Si el servidor rechaza la conexión en el puerto 8767, la guía de arranque está en `docs/momento_aclaracion_v6.md` dentro del proyecto remoto.

## Qué se conserva

Los archivos guardados en la carpeta remota permanecen en el disco del servidor al cerrar VS Code, desconectar la VPN o apagar la Mac. El chat guarda sus conversaciones en `outputs/final_food_chats/`. El navegador conserva la referencia para reabrirlas, por lo que una Mac nueva puede mostrar el historial vacío. La política institucional de respaldos queda por confirmar.

El repositorio público contiene el código y los adaptadores de la tesis. Guardar un archivo en el servidor y publicarlo en GitHub son acciones independientes. Guarda los datos de tu cuenta en la configuración SSH de tu Mac. [Mesa y Memoria en GitHub](https://github.com/kalech7/mesa-y-memoria)
